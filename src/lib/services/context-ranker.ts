/**
 * Context Ranker Service - Modular context selection for AI models
 *
 * This service decides what context to provide to AI models (like Gemma)
 * by ranking and selecting the most relevant content based on query embeddings.
 */

import type { OllamaEmbedding } from '$lib/ai/types';
import { postgres } from '$lib/database/postgres';

export interface ContextRankerConfig {
  topK?: number;
  embeddingModel?: string;
  maxContextLength?: number;
  similarityThreshold?: number;
  includeMetadata?: boolean;
  cacheTTL?: number;
}

export interface RankedContext {
  content: string;
  similarity: number;
  metadata?: unknown;
  source?: string;
}

export interface ContextRankingResult {
  contexts: RankedContext[];
  totalTokens: number;
  processingTime: number;
  query: string;
}

class ContextRanker {
  private config: Required<ContextRankerConfig>;
  private cache: Map<string, { result: ContextRankingResult; timestamp: number }>;
  private ollamaEndpoint: string;

  constructor(config: ContextRankerConfig = {}) {
    this.config = {
      topK: config.topK ?? 5,
      embeddingModel: config.embeddingModel ?? 'nomic-embed-text',
      maxContextLength: config.maxContextLength ?? 4000,
      similarityThreshold: config.similarityThreshold ?? 0.3,
      includeMetadata: config.includeMetadata ?? true,
      cacheTTL: config.cacheTTL ?? 300000 // 5 minutes
    };

    this.cache = new Map();
    this.ollamaEndpoint = process.env.OLLAMA_ENDPOINT || 'http://localhost:11434';
  }

  /**
   * Main context ranking function
   * Decides what context the AI model should see based on query relevance
   */
  async context_ranker(
    query: string,
    options: Partial<ContextRankerConfig> = {}
  ): Promise<ContextRankingResult> {
    const startTime = performance.now();
    const mergedConfig = { ...this.config, ...options };

    // Check cache first
    const cacheKey = this.generateCacheKey(query, mergedConfig);
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      return cached;
    }

    try {
      // Step 1: Embed user query
      console.log(`🔍 Generating embedding for query: "${query.substring(0, 50)}..."`);
      const queryEmbedding = await this.embedWithOllama(query);

      // Step 2: Perform ANN search (pgvector)
      console.log(`🎯 Searching for top ${mergedConfig.topK} most relevant contexts`);
      const rankedContexts = await this.searchSimilarContent(
        queryEmbedding,
        mergedConfig
      );

      // Step 3: Filter by similarity threshold
      const filteredContexts = rankedContexts.filter(
        ctx => ctx.similarity >= mergedConfig.similarityThreshold
      );

      // Step 4: Truncate to max context length
      const optimizedContexts = this.optimizeContextLength(
        filteredContexts,
        mergedConfig.maxContextLength
      );

      const result: ContextRankingResult = {
        contexts: optimizedContexts,
        totalTokens: this.estimateTokenCount(optimizedContexts),
        processingTime: performance.now() - startTime,
        query
      };

      // Cache the result
      this.setInCache(cacheKey, result);

      console.log(`✅ Context ranking complete: ${result.contexts.length} contexts, ${result.totalTokens} tokens, ${result.processingTime.toFixed(2)}ms`);

      return result;

    } catch (error) {
      console.error('❌ Context ranking failed:', error);
      throw new Error(`Context ranking failed: ${error.message}`);
    }
  }

  /**
   * Generate embeddings using Ollama
   */
  private async embedWithOllama(text: string): Promise<number[]> {
    try {
      const response = await fetch(`${this.ollamaEndpoint}/api/embeddings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: this.config.embeddingModel,
          prompt: text
        })
      });

      if (!response.ok) {
        throw new Error(`Ollama embedding failed: ${response.statusText}`);
      }

      const data: OllamaEmbedding = await response.json();
      return data.embedding;

    } catch (error) {
      console.error('❌ Embedding generation failed:', error);
      throw error;
    }
  }

  /**
   * Search for similar content using pgvector
   */
  private async searchSimilarContent(
    queryEmbedding: number[],
    config: Required<ContextRankerConfig>
  ): Promise<RankedContext[]> {
    try {
      // Use pgvector for efficient similarity search
      const embeddingVector = `[${queryEmbedding.join(',')}]`;

      const query = `
        SELECT
          content,
          title,
          document_type,
          metadata,
          file_name,
          practice_area,
          1 - (content_embedding <=> $1::vector) AS similarity
        FROM legal_documents
        WHERE content_embedding IS NOT NULL
          AND content IS NOT NULL
          AND length(content) > 50
        ORDER BY content_embedding <=> $1::vector
        LIMIT $2
      `;

      const { rows } = await postgres.query(query, [embeddingVector, config.topK]);

      return rows.map(row => ({
        content: row.content,
        similarity: parseFloat(row.similarity) || 0,
        metadata: config.includeMetadata ? {
          title: row.title,
          documentType: row.document_type,
          fileName: row.file_name,
          practiceArea: row.practice_area,
          ...row.metadata
        } : undefined,
        source: row.title || row.file_name || 'Unknown'
      }));

    } catch (error) {
      console.error('❌ pgvector search failed:', error);
      throw error;
    }
  }

  /**
   * Optimize context length to fit within token limits
   */
  private optimizeContextLength(
    contexts: RankedContext[],
    maxLength: number
  ): RankedContext[] {
    const optimized: RankedContext[] = [];
    let currentLength = 0;

    for (const context of contexts) {
      const contextLength = context.content.length;

      if (currentLength + contextLength <= maxLength) {
        optimized.push(context);
        currentLength += contextLength;
      } else {
        // Truncate the last context to fit
        const remainingLength = maxLength - currentLength;
        if (remainingLength > 100) { // Only include if meaningful length remains
          optimized.push({
            ...context,
            content: context.content.substring(0, remainingLength) + '...'
          });
        }
        break;
      }
    }

    return optimized;
  }

  /**
   * Estimate token count for contexts
   */
  private estimateTokenCount(contexts: RankedContext[]): number {
    // Rough estimation: ~4 characters per token
    const totalChars = contexts.reduce((sum, ctx) => sum + ctx.content.length, 0);
    return Math.ceil(totalChars / 4);
  }

  /**
   * Format contexts for AI model consumption
   */
  formatContextsForAI(result: ContextRankingResult): string {
    if (result.contexts.length === 0) {
      return "No relevant context found.";
    }

    return result.contexts
      .map((ctx, index) => {
        let formatted = `--- Context ${index + 1} (similarity: ${ctx.similarity.toFixed(3)}) ---\n`;

        if (ctx.metadata?.title) {
          formatted += `Title: ${ctx.metadata.title}\n`;
        }

        if (ctx.source) {
          formatted += `Source: ${ctx.source}\n`;
        }

        formatted += `Content:\n${ctx.content}\n`;

        return formatted;
      })
      .join('\n');
  }

  /**
   * Get summary statistics about context ranking
   */
  getContextStats(result: ContextRankingResult) {
    return {
      contextCount: result.contexts.length,
      averageSimilarity: result.contexts.reduce((sum, ctx) => sum + ctx.similarity, 0) / result.contexts.length,
      maxSimilarity: Math.max(...result.contexts.map(ctx => ctx.similarity)),
      minSimilarity: Math.min(...result.contexts.map(ctx => ctx.similarity)),
      totalTokens: result.totalTokens,
      processingTime: result.processingTime,
      sources: [...new Set(result.contexts.map(ctx => ctx.source))],
      documentTypes: [...new Set(result.contexts.map(ctx => ctx.metadata?.documentType).filter(Boolean))]
    };
  }

  /**
   * Cache management
   */
  private generateCacheKey(query: string, config: Required<ContextRankerConfig>): string {
    const configHash = JSON.stringify({
      topK: config.topK,
      model: config.embeddingModel,
      threshold: config.similarityThreshold
    });
    return `${query}:${btoa(configHash)}`;
  }

  private getFromCache(key: string): ContextRankingResult | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < this.config.cacheTTL) {
      console.log('📋 Using cached context ranking result');
      return cached.result;
    }
    return null;
  }

  private setInCache(key: string, result: ContextRankingResult): void {
    this.cache.set(key, {
      result,
      timestamp: Date.now()
    });

    // Basic cache cleanup
    if (this.cache.size > 100) {
      const oldestKey = this.cache.keys().next().value;
      this.cache.delete(oldestKey);
    }
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache.clear();
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<ContextRankerConfig>): void {
    this.config = { ...this.config, ...newConfig };
    this.clearCache(); // Clear cache when config changes
  }
}

// Export singleton instance and class
export const contextRanker = new ContextRanker();
export { ContextRanker };

// Export utility functions for use in other modules
export async function rankContext(
  query: string,
  options?: Partial<ContextRankerConfig>
): Promise<ContextRankingResult> {
  return contextRanker.context_ranker(query, options);
}

export function formatContextForAI(result: ContextRankingResult): string {
  return contextRanker.formatContextsForAI(result);
}

export function getContextStatistics(result: ContextRankingResult) {
  return contextRanker.getContextStats(result);
}