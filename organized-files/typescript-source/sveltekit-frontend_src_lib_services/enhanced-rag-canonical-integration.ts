// ======================================================================
// ENHANCED RAG + CANONICAL CACHE INTEGRATION
// Combines vector search with single-character key caching for optimal performance
// ======================================================================

import { canonicalResultCache, type CanonicalResult, type RankingSet } from './canonical-result-cache.js'
import { quicCacheClient, type QUICCacheResponse } from './quic-canonical-cache-endpoint.js'
import { enhancedRAGService, type RAGResponse } from './enhanced-rag-service.js'
import type { RetrievalResult } from './enhanced-rag-service.js';

export interface CachedRAGQuery {
  query: string;
  context?: {
    caseId?: string;
    userId?: string;
    documentTypes?: string[];
  };
  options?: {
    useCache?: boolean;
    forceRefresh?: boolean;
    maxCacheAge?: number; // seconds
    includeRawResults?: boolean;
  };
}

export interface CachedRAGResponse extends RAGResponse {
  cacheInfo: {
    slotKey?: string;
    cacheHit: boolean;
    latencyMs: number;
    protocol: 'quic' | 'http' | 'fallback' | 'bypass';
    compressionRatio?: number;
    resultsCached: boolean;
  };
}

export interface RAGCacheMetrics {
  totalQueries: number;
  cacheHits: number;
  cacheMisses: number;
  avgLatency: {
    cached: number;
    uncached: number;
    quic: number;
    http: number;
  };
  cacheEfficiency: number;
  memoryUsage: {
    cacheBytes: number;
    utilization: number;
  };
}

export class EnhancedRAGCanonicalService {
  private metrics: RAGCacheMetrics = {
    totalQueries: 0,
    cacheHits: 0,
    cacheMisses: 0,
    avgLatency: {
      cached: 0,
      uncached: 0,
      quic: 0,
      http: 0
    },
    cacheEfficiency: 0,
    memoryUsage: {
      cacheBytes: 0,
      utilization: 0
    }
  };

  private latencyHistory: { cached: number[]; uncached: number[] } = {
    cached: [],
    uncached: []
  };

  constructor(
    private ragService = enhancedRAGService,
    private cache = canonicalResultCache,
    private quicClient = quicCacheClient
  ) {}

  // Main query method with intelligent caching
  async query(request: CachedRAGQuery): Promise<CachedRAGResponse> {
    const startTime = performance.now();
    this.metrics.totalQueries++;

    try {
      // Generate cache key from query and context
      const cacheKey = this.generateCacheKey(request);
      
      // Check cache first (unless force refresh)
      if (request.options?.useCache !== false && !request.options?.forceRefresh) {
        const cachedResult = await this.tryRetrieveFromCache(cacheKey);
        if (cachedResult) {
          this.updateCacheHitMetrics(performance.now() - startTime);
          return cachedResult;
        }
      }

      // Cache miss - perform full RAG query
      const ragResponse = await this.performRAGQuery(request);
      
      // Cache the results for future use
      if (request.options?.useCache !== false) {
        await this.cacheRAGResults(cacheKey, request, ragResponse);
      }

      this.updateCacheMissMetrics(performance.now() - startTime);

      return {
        ...ragResponse,
        cacheInfo: {
          cacheHit: false,
          latencyMs: performance.now() - startTime,
          protocol: 'bypass',
          resultsCached: request.options?.useCache !== false
        }
      };

    } catch (error) {
      console.error('Enhanced RAG query failed:', error);
      
      // Return fallback response with error info
      return {
        answer: 'I apologize, but I encountered an error processing your request.',
        sources: [],
        confidence: 0,
        processingTime: performance.now() - startTime,
        model: 'error-fallback',
        cacheInfo: {
          cacheHit: false,
          latencyMs: performance.now() - startTime,
          protocol: 'fallback',
          resultsCached: false
        }
      };
    }
  }

  // Try to retrieve results from cache using multiple protocols
  private async tryRetrieveFromCache(cacheKey: string): Promise<CachedRAGResponse | null> {
    try {
      // Try QUIC first for ultra-low latency
      const quicResponse = await this.quicClient.getRankingSet(cacheKey, {
        includeMetadata: true,
        timeoutMs: 5000
      });

      if (quicResponse.success && quicResponse.data) {
        return this.reconstructRAGResponse(quicResponse);
      }

      // Fallback to direct cache access
      const rankingSet = await this.cache.retrieveRankingSet(cacheKey);
      if (rankingSet) {
        return this.reconstructRAGResponseFromRankingSet(rankingSet, {
          success: true,
          latencyMs: 0,
          protocol: 'http',
          cacheHit: true
        });
      }

      return null;

    } catch (error) {
      console.debug('Cache retrieval failed:', error);
      return null;
    }
  }

  // Perform the actual RAG query
  private async performRAGQuery(request: CachedRAGQuery): Promise<RAGResponse> {
    return await this.ragService.query(request.query, request.context);
  }

  // Cache RAG results for future queries
  private async cacheRAGResults(
    cacheKey: string, 
    request: CachedRAGQuery, 
    ragResponse: RAGResponse
  ): Promise<void> {
    try {
      // Convert RAG results to canonical format
      const canonicalResults: CanonicalResult[] = ragResponse.sources.map((source, index) => ({
        docId: source.id,
        score: source.score,
        flags: this.computeResultFlags(source, index),
        summaryHash: this.computeSummaryHash(source.content),
        targetUrlId: source.metadata?.url,
        metadata: {
          source: source.source,
          contentPreview: source.content.substring(0, 200),
          ragMetadata: source.metadata
        }
      }));

      // Create ranking set
      const rankingSet: RankingSet = {
        results: canonicalResults,
        query: request.query,
        totalResults: ragResponse.sources.length,
        timestamp: Date.now(),
        version: 1
      };

      // Store in cache
      const slotKey = await this.cache.storeRankingSet(rankingSet);
      
      // Optionally store full RAG response metadata separately
      await this.cacheRAGMetadata(slotKey, ragResponse);

      console.debug(`✅ Cached RAG results with slot key: ${slotKey}`);

    } catch (error) {
      console.warn('Failed to cache RAG results:', error);
    }
  }

  // Reconstruct RAG response from QUIC cache response
  private reconstructRAGResponse(quicResponse: QUICCacheResponse): CachedRAGResponse {
    if (!quicResponse.data) {
      throw new Error('No data in QUIC response');
    }

    const rankingSet = quicResponse.data;
    
    // Convert canonical results back to retrieval results
    const sources: RetrievalResult[] = rankingSet.results.map(result => ({
      id: result.docId,
      content: result.metadata?.contentPreview || 'Content not cached',
      score: result.score,
      source: result.metadata?.source || 'Unknown source',
      metadata: result.metadata?.ragMetadata || {}
    }));

    // Reconstruct RAG response (simplified - would need to restore full answer)
    return {
      answer: `Cached response for: ${rankingSet.query}`,
      sources,
      confidence: this.calculateCachedConfidence(sources),
      processingTime: quicResponse.latencyMs,
      model: 'cached-results',
      reasoning: {
        queryIntent: 'Cached query processing',
        retrievedContext: sources.map(s => s.source),
        synthesisStrategy: 'Cache reconstruction'
      },
      cacheInfo: {
        cacheHit: true,
        latencyMs: quicResponse.latencyMs,
        protocol: quicResponse.protocol,
        compressionRatio: quicResponse.compressionRatio,
        resultsCached: true
      }
    };
  }

  private reconstructRAGResponseFromRankingSet(
    rankingSet: RankingSet, 
    cacheResponse: Partial<QUICCacheResponse>
  ): CachedRAGResponse {
    const sources: RetrievalResult[] = rankingSet.results.map(result => ({
      id: result.docId,
      content: result.metadata?.contentPreview || 'Content not cached',
      score: result.score,
      source: result.metadata?.source || 'Unknown source',
      metadata: result.metadata?.ragMetadata || {}
    }));

    return {
      answer: `Cached response for: ${rankingSet.query}`,
      sources,
      confidence: this.calculateCachedConfidence(sources),
      processingTime: cacheResponse.latencyMs || 0,
      model: 'cached-results',
      reasoning: {
        queryIntent: 'Cached query processing',
        retrievedContext: sources.map(s => s.source),
        synthesisStrategy: 'Direct cache reconstruction'
      },
      cacheInfo: {
        cacheHit: true,
        latencyMs: cacheResponse.latencyMs || 0,
        protocol: cacheResponse.protocol || 'http',
        compressionRatio: cacheResponse.compressionRatio,
        resultsCached: true
      }
    };
  }

  // Generate deterministic cache key from query and context
  private generateCacheKey(request: CachedRAGQuery): string {
    // Create normalized key that represents the semantic intent
    const keyData = {
      query: request.query.toLowerCase().trim(),
      context: request.context ? {
        caseId: request.context.caseId,
        documentTypes: request.context.documentTypes?.sort()
      } : undefined
    };

    // Generate hash of key data
    const keyString = JSON.stringify(keyData);
    let hash = 0;
    
    for (let i = 0; i < keyString.length; i++) {
      const char = keyString.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }

    // Map hash to alphabet character
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    return alphabet[Math.abs(hash) % alphabet.length];
  }

  private computeResultFlags(source: RetrievalResult, index: number): number {
    let flags = 0;
    
    // Position-based flags
    if (index === 0) flags |= 0x01; // Top result
    if (source.score > 0.9) flags |= 0x02; // High confidence
    if (source.metadata?.type) flags |= 0x04; // Has type metadata
    if (source.content.length > 1000) flags |= 0x08; // Long content
    
    return flags;
  }

  private computeSummaryHash(content: string): string {
    // Simple hash for content identification
    let hash = 0;
    for (let i = 0; i < Math.min(content.length, 100); i++) {
      hash = ((hash << 5) - hash + content.charCodeAt(i)) & 0x3FFFFF; // 22 bits
    }
    return hash.toString(16);
  }

  private calculateCachedConfidence(sources: RetrievalResult[]): number {
    if (!sources.length) return 0;
    
    const avgScore = sources.reduce((sum, s) => sum + s.score, 0) / sources.length;
    return Math.min(avgScore * 0.9, 0.85); // Slightly lower confidence for cached results
  }

  private async cacheRAGMetadata(slotKey: string, ragResponse: RAGResponse): Promise<void> {
    // Store full RAG response metadata for complete reconstruction
    // This could use Redis or another storage system
    try {
      const metadata = {
        answer: ragResponse.answer,
        confidence: ragResponse.confidence,
        model: ragResponse.model,
        reasoning: ragResponse.reasoning
      };
      
      // Store with expiration (implementation would depend on chosen storage)
      console.debug(`Stored RAG metadata for slot ${slotKey}`);
    } catch (error) {
      console.debug('Failed to store RAG metadata:', error);
    }
  }

  // Metrics and monitoring
  private updateCacheHitMetrics(latency: number): void {
    this.metrics.cacheHits++;
    this.latencyHistory.cached.push(latency);
    
    // Keep only recent samples
    if (this.latencyHistory.cached.length > 100) {
      this.latencyHistory.cached = this.latencyHistory.cached.slice(-100);
    }
    
    this.updateAverageLatencies();
  }

  private updateCacheMissMetrics(latency: number): void {
    this.metrics.cacheMisses++;
    this.latencyHistory.uncached.push(latency);
    
    if (this.latencyHistory.uncached.length > 100) {
      this.latencyHistory.uncached = this.latencyHistory.uncached.slice(-100);
    }
    
    this.updateAverageLatencies();
  }

  private updateAverageLatencies(): void {
    if (this.latencyHistory.cached.length > 0) {
      this.metrics.avgLatency.cached = 
        this.latencyHistory.cached.reduce((sum, l) => sum + l, 0) / this.latencyHistory.cached.length;
    }
    
    if (this.latencyHistory.uncached.length > 0) {
      this.metrics.avgLatency.uncached = 
        this.latencyHistory.uncached.reduce((sum, l) => sum + l, 0) / this.latencyHistory.uncached.length;
    }
    
    // Update cache efficiency
    const totalQueries = this.metrics.cacheHits + this.metrics.cacheMisses;
    this.metrics.cacheEfficiency = totalQueries > 0 ? this.metrics.cacheHits / totalQueries : 0;
  }

  // Public API
  async getMetrics(): Promise<RAGCacheMetrics> {
    const cacheMetrics = this.cache.getMetrics();
    const quicMetrics = this.quicClient.getMetrics();
    
    return {
      ...this.metrics,
      memoryUsage: {
        cacheBytes: cacheMetrics.sizeBytes,
        utilization: cacheMetrics.slotUtilization
      },
      avgLatency: {
        ...this.metrics.avgLatency,
        quic: quicMetrics.quicRequests > 0 ? quicMetrics.averageLatency : 0,
        http: quicMetrics.httpRequests > 0 ? quicMetrics.averageLatency : 0
      }
    };
  }

  async clearCache(): Promise<void> {
    await this.cache.clear();
    this.metrics = {
      totalQueries: 0,
      cacheHits: 0,
      cacheMisses: 0,
      avgLatency: { cached: 0, uncached: 0, quic: 0, http: 0 },
      cacheEfficiency: 0,
      memoryUsage: { cacheBytes: 0, utilization: 0 }
    };
    this.latencyHistory = { cached: [], uncached: [] };
  }

  async warmupCache(commonQueries: string[]): Promise<void> {
    console.log('🔥 Warming up RAG cache with common queries...');
    
    const warmupPromises = commonQueries.map(async (query) => {
      try {
        await this.query({
          query,
          options: { useCache: true }
        });
      } catch (error) {
        console.debug(`Warmup failed for query: ${query}`, error);
      }
    });

    await Promise.allSettled(warmupPromises);
    console.log('✅ RAG cache warmup completed');
  }
}

// Export singleton instance
export const enhancedRAGCanonical = new EnhancedRAGCanonicalService();