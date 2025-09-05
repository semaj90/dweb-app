/**
 * Redis Integration Service for Legal AI Inference Pipeline
 * Connects all pipeline components with unified caching layer
 */

import { redisCacheManager, type CacheHealth } from '../../../redis-config/redis-cache-helpers';
import { createHash } from 'crypto';

// Integration types
interface TokenizationRequest {
  text: string;
  model: string;
  userId?: string;
}

interface EmbeddingRequest {
  texts: string[];
  model: string;
  userId?: string;
  metadata?: Record<string, any>;
}

interface InferenceRequest {
  prompt: string;
  model: string;
  temperature: number;
  maxTokens: number;
  userId?: string;
}

// Legal document processing types
interface LegalDocument {
  id: string;
  caseId?: string;
  content: string;
  documentType: 'contract' | 'evidence' | 'brief' | 'citation';
  metadata: Record<string, any>;
}

export class RedisInferenceIntegration {
  private static instance: RedisInferenceIntegration;
  
  static getInstance(): RedisInferenceIntegration {
    if (!RedisInferenceIntegration.instance) {
      RedisInferenceIntegration.instance = new RedisInferenceIntegration();
    }
    return RedisInferenceIntegration.instance;
  }

  async initialize(): Promise<boolean> {
    try {
      const connected = await redisCacheManager.connect();
      if (connected) {
        console.log('✅ Redis inference integration initialized');
        await this.warmupCache();
      }
      return connected;
    } catch (error) {
      console.error('❌ Redis initialization failed:', error);
      return false;
    }
  }

  // Tokenization caching layer
  async cacheTokenization(request: TokenizationRequest, tokens: number[]): Promise<void> {
    try {
      await redisCacheManager.cacheTokens(request.text, tokens, request.model);
      
      // Track usage statistics
      if (request.userId) {
        await this.incrementUserStats(request.userId, 'tokenizations');
      }
    } catch (error) {
      console.error('Tokenization cache error:', error);
    }
  }

  async getCachedTokenization(request: TokenizationRequest): Promise<number[] | null> {
    try {
      return await redisCacheManager.getTokens(request.text, request.model);
    } catch (error) {
      console.error('Tokenization cache retrieval error:', error);
      return null;
    }
  }

  // Embedding caching layer
  async batchCacheEmbeddings(request: EmbeddingRequest, embeddings: number[][]): Promise<void> {
    try {
      const items = request.texts.map((text, index) => ({
        text,
        embedding: embeddings[index],
        metadata: { ...request.metadata, userId: request.userId }
      }));

      await redisCacheManager.batchCacheEmbeddings(items, request.model);
      
      // Track usage
      if (request.userId) {
        await this.incrementUserStats(request.userId, 'embeddings', embeddings.length);
      }
    } catch (error) {
      console.error('Embedding cache error:', error);
    }
  }

  async getBatchCachedEmbeddings(request: EmbeddingRequest): Promise<Array<{ text: string; embedding: number[] | null }>> {
    try {
      return await redisCacheManager.batchGetEmbeddings(request.texts, request.model);
    } catch (error) {
      console.error('Embedding cache retrieval error:', error);
      return request.texts.map(text => ({ text, embedding: null }));
    }
  }

  // Inference result caching
  async cacheInferenceResult(request: InferenceRequest, result: any): Promise<void> {
    try {
      await redisCacheManager.cacheInferenceResult(
        request.prompt,
        result,
        request.model,
        request.temperature
      );

      // Track token usage
      if (request.userId && result.tokensUsed) {
        await this.incrementUserStats(request.userId, 'tokens_used', result.tokensUsed);
      }
    } catch (error) {
      console.error('Inference cache error:', error);
    }
  }

  async getCachedInferenceResult(request: InferenceRequest): Promise<any> {
    try {
      return await redisCacheManager.getInferenceResult(
        request.prompt,
        request.model,
        request.temperature
      );
    } catch (error) {
      console.error('Inference cache retrieval error:', error);
      return null;
    }
  }

  // Legal document analysis caching
  async cacheLegalDocumentAnalysis(
    document: LegalDocument,
    analysisType: string,
    analysis: any
  ): Promise<void> {
    try {
      await redisCacheManager.cacheLegalAnalysis(
        document.id,
        analysisType,
        {
          ...analysis,
          document_type: document.documentType,
          content_hash: this.hashContent(document.content)
        },
        document.caseId
      );
    } catch (error) {
      console.error('Legal analysis cache error:', error);
    }
  }

  async getCachedLegalDocumentAnalysis(
    documentId: string,
    analysisType: string,
    caseId?: string
  ): Promise<any> {
    try {
      return await redisCacheManager.getLegalAnalysis(documentId, analysisType, caseId);
    } catch (error) {
      console.error('Legal analysis cache retrieval error:', error);
      return null;
    }
  }

  // Vector similarity caching
  async cacheSimilaritySearch(
    queryVector: number[],
    results: any[],
    options: {
      caseId?: string;
      documentType?: string;
      threshold?: number;
      limit?: number;
    } = {}
  ): Promise<void> {
    try {
      await redisCacheManager.cacheSimilarityResults(queryVector, results, options);
    } catch (error) {
      console.error('Similarity cache error:', error);
    }
  }

  async getCachedSimilaritySearch(
    queryVector: number[],
    options: {
      caseId?: string;
      documentType?: string;
      threshold?: number;
      limit?: number;
    } = {}
  ): Promise<any[] | null> {
    try {
      return await redisCacheManager.getSimilarityResults(queryVector, options);
    } catch (error) {
      console.error('Similarity cache retrieval error:', error);
      return null;
    }
  }

  // Prefetch management for legal workflows
  async prefetchLegalQueries(caseId: string, documentTypes: string[]): Promise<void> {
    try {
      const commonQueries = [
        `Analyze key clauses in ${documentTypes.join(', ')} documents for case ${caseId}`,
        `Identify potential risks in case ${caseId} documents`,
        `Extract important dates and deadlines from case ${caseId}`,
        `Summarize evidence for case ${caseId}`
      ];

      for (const query of commonQueries) {
        await redisCacheManager.cachePrefetch(query, null, 'low');
      }
    } catch (error) {
      console.error('Legal prefetch error:', error);
    }
  }

  // Pipeline health monitoring
  async getPipelineHealth(): Promise<{
    redis: CacheHealth;
    pipeline: {
      tokenization_cache_hit_rate: number;
      embedding_cache_hit_rate: number;
      inference_cache_hit_rate: number;
      total_requests_served: number;
      average_response_time: number;
    };
  }> {
    try {
      const redisHealth = await redisCacheManager.getCacheHealth();
      const stats = await redisCacheManager.getCacheStats();

      // Calculate hit rates
      const tokenHitRate = this.calculateHitRate(stats, 'tokens');
      const embeddingHitRate = this.calculateHitRate(stats, 'embeddings');
      const inferenceHitRate = this.calculateHitRate(stats, 'results');

      return {
        redis: redisHealth,
        pipeline: {
          tokenization_cache_hit_rate: tokenHitRate,
          embedding_cache_hit_rate: embeddingHitRate,
          inference_cache_hit_rate: inferenceHitRate,
          total_requests_served: (stats['cache:tokens:hits'] || 0) + 
                                 (stats['cache:embeddings:hits'] || 0) + 
                                 (stats['cache:results:hits'] || 0),
          average_response_time: 0 // Would be calculated from timing metrics
        }
      };
    } catch (error) {
      console.error('Health check error:', error);
      throw error;
    }
  }

  // Cache warming for legal AI models
  private async warmupCache(): Promise<void> {
    try {
      // Common legal terms for tokenization cache
      const legalTerms = [
        'contract', 'liability', 'defendant', 'plaintiff', 'evidence',
        'jurisdiction', 'precedent', 'statute of limitations', 'due process'
      ];

      // Warm up tokenization cache
      for (const term of legalTerms) {
        await this.cacheTokenization(
          { text: term, model: 'legal-tokenizer' },
          [1, 2, 3] // Mock tokens for warmup
        );
      }

      console.log('✅ Cache warmup completed');
    } catch (error) {
      console.error('Cache warmup failed:', error);
    }
  }

  // User statistics tracking
  private async incrementUserStats(userId: string, metric: string, count: number = 1): Promise<void> {
    try {
      const key = `user:${userId}:${metric}`;
      await redisCacheManager['redis'].incrby(key, count);
      await redisCacheManager['redis'].expire(key, 30 * 24 * 60 * 60); // 30 days TTL
    } catch (error) {
      console.error('User stats error:', error);
    }
  }

  private calculateHitRate(stats: Record<string, number>, type: string): number {
    const hits = stats[`cache:${type}:hits`] || 0;
    const misses = stats[`cache:${type}:misses`] || 0;
    const total = hits + misses;
    return total > 0 ? (hits / total) * 100 : 0;
  }

  private hashContent(content: string): string {
    return createHash('sha256').update(content).digest('hex').slice(0, 16);
  }

  // Cleanup methods
  async clearUserCache(userId: string): Promise<void> {
    try {
      await redisCacheManager.clearCache(`*:${userId}:*`);
    } catch (error) {
      console.error('User cache clear error:', error);
    }
  }

  async clearCaseCache(caseId: string): Promise<void> {
    try {
      await redisCacheManager.clearCache(`*:${caseId}:*`);
    } catch (error) {
      console.error('Case cache clear error:', error);
    }
  }

  async disconnect(): Promise<void> {
    await redisCacheManager.disconnect();
  }
}

// Export singleton instance
export const redisIntegration = RedisInferenceIntegration.getInstance();

// Initialize on module load in server environment
if (typeof window === 'undefined') {
  redisIntegration.initialize().catch(console.error);
}