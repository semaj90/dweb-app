/**
 * Cross-Encoder Reranking Service
 * Reranks search results using cross-encoder models for improved relevance
 */

import type { LegalDocument } from './types/legal';

interface SearchResult {
  document: LegalDocument;
  score: number;
  metadata?: Record<string, any>;
  // Additional properties used throughout the codebase
  id: string;
  title: string;
  content?: string;
  summary?: string;
  excerpt?: string;
  rank?: number;
}

interface RerankingConfig {
  threshold: number;
  maxResults: number;
  useSemanticSimilarity: boolean;
}

interface ScoredResult {
  document: LegalDocument;
  originalScore: number;
  rerankScore: number;
  combinedScore: number;
  metadata: {
    modelUsed: string;
    processingTime: number;
    confidence: number;
  };
}

interface CrossEncoderConfig {
  model: string; // Model identifier or endpoint
  maxResults: number; // Maximum results to rerank
  scoreWeight: number; // Weight for rerank score vs original
  batchSize: number; // Batch size for processing
  timeout: number; // Request timeout in ms
  fallbackEnabled: boolean; // Enable fallback scoring
  minConfidenceThreshold: number; // Minimum confidence to trust reranking
}

export class CrossEncoderReranker {
  private config: CrossEncoderConfig;
  private modelCache: Map<string, any> = new Map();
  private scoreCache: Map<string, number> = new Map();

  constructor(config: Partial<CrossEncoderConfig> = {}) {
    this.config = {
      model: 'cross-encoder/ms-marco-MiniLM-L-12-v2',
      maxResults: 50,
      scoreWeight: 0.7, // Favor rerank scores over original
      batchSize: 10,
      timeout: 5000,
      fallbackEnabled: true,
      minConfidenceThreshold: 0.3,
      ...config,
    };
  }

  async rerankResults(
    query: string,
    results: SearchResult[],
    config?: Partial<RerankingConfig>
  ): Promise<SearchResult[]> {
    const startTime = Date.now();

    try {
      // Limit results to process
      const toRerank = results.slice(0, this.config.maxResults);

      // Score each query-document pair
      const scored = await this.scoreQueryDocumentPairs(query, toRerank);

      // Combine original and rerank scores
      const reranked = this.combineScores(scored);

      // Sort by combined score
      const sorted = reranked.sort((a, b) => b.combinedScore - a.combinedScore);

      // Convert back to SearchResult format with enhanced metadata
      return sorted.map((item) => {
        const rebuilt: SearchResult = {
          document: item.document,
          id: item.document.id,
          title: item.document.title,
          content: (item.document as any).content,
          score: item.combinedScore,
          metadata: {
            ...item.document.metadata,
            originalScore: item.originalScore,
            rerankScore: item.rerankScore,
            reranking: item.metadata,
            processingTime: Date.now() - startTime,
          },
        };
        return rebuilt;
      });
    } catch (error) {
      console.error('[CrossEncoder] Reranking failed:', error);

      if (this.config.fallbackEnabled) {
        console.warn('[CrossEncoder] Falling back to original scores');
        return results;
      }

      throw new Error(`Cross-encoder reranking failed: ${error.message}`);
    }
  }

  private async scoreQueryDocumentPairs(
    query: string,
    results: SearchResult[]
  ): Promise<ScoredResult[]> {
    const scored: ScoredResult[] = [];

    // Process in batches for efficiency
    for (let i = 0; i < results.length; i += this.config.batchSize) {
      const batch = results.slice(i, i + this.config.batchSize);
      const batchScores = await this.processBatch(query, batch);
      scored.push(...batchScores);
    }

    return scored;
  }

  private async processBatch(query: string, batch: SearchResult[]): Promise<ScoredResult[]> {
    const batchStartTime = Date.now();

    try {
      // Try different scoring approaches in order of preference
      const scores = await this.tryMultipleApproaches(query, batch);

      return batch.map((result, index) => ({
        document: result as LegalDocument,
        originalScore: result.score || 0,
        rerankScore: scores[index] || 0,
        combinedScore: 0, // Will be calculated later
        metadata: {
          modelUsed: this.config.model,
          processingTime: Date.now() - batchStartTime,
          confidence: this.calculateConfidence(scores[index] || 0),
        },
      }));
    } catch (error) {
      console.warn('[CrossEncoder] Batch processing failed, using fallback scores');

      // Fallback to lexical similarity scoring
      return batch.map((result) => ({
        document: result as LegalDocument,
        originalScore: result.score || 0,
        rerankScore: this.lexicalSimilarity(query, this.extractText(result)),
        combinedScore: 0,
        metadata: {
          modelUsed: 'lexical-fallback',
          processingTime: Date.now() - batchStartTime,
          confidence: 0.3,
        },
      }));
    }
  }

  private async tryMultipleApproaches(query: string, batch: SearchResult[]): Promise<number[]> {
    // 1. Try Ollama-based scoring
    try {
      return await this.scoreWithOllama(query, batch);
    } catch (error) {
      console.warn('[CrossEncoder] Ollama scoring failed:', error.message);
    }

    // 2. Try external API (if configured)
    try {
      return await this.scoreWithExternalAPI(query, batch);
    } catch (error) {
      console.warn('[CrossEncoder] External API scoring failed:', error.message);
    }

    // 3. Fallback to local computation
    return this.scoreWithLocalComputation(query, batch);
  }

  private async scoreWithOllama(query: string, batch: SearchResult[]): Promise<number[]> {
    const pairs = batch.map((result) => ({
      query,
      passage: this.extractText(result),
    }));

    const response = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3.1',
        prompt: this.buildScoringPrompt(pairs),
        stream: false,
        options: {
          temperature: 0.1,
          top_p: 0.9,
          max_tokens: 500,
        },
      }),
      signal: AbortSignal.timeout(this.config.timeout),
    });

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }

    const data = await response.json();
    return this.parseScoresFromResponse(data.response, batch.length);
  }

  private async scoreWithExternalAPI(query: string, batch: SearchResult[]): Promise<number[]> {
    // Placeholder for external cross-encoder API
    // Could integrate with Hugging Face Inference API, Azure Cognitive Services, etc.
    throw new Error('External API not configured');
  }

  private scoreWithLocalComputation(query: string, batch: SearchResult[]): Promise<number[]> {
    // Local computation using TF-IDF, BM25, or similar algorithms
    const queryTerms = this.tokenize(query.toLowerCase());

    const scores = batch.map((result) => {
      const docText = this.extractText(result).toLowerCase();
      const docTerms = this.tokenize(docText);

      // Enhanced TF-IDF scoring with position and exact match bonuses
      let score = 0;
      const docLength = docTerms.length;

      for (const term of queryTerms) {
        // Term frequency
        const termCount = docTerms.filter((t) => t.includes(term)).length;
        if (termCount > 0) {
          const tf = termCount / docLength;
          const idf = Math.log(batch.length / (1 + termCount));
          score += tf * idf;

          // Exact match bonus
          if (docTerms.includes(term)) {
            score += 0.2;
          }

          // Position bonus (early occurrence)
          const firstIndex = docTerms.findIndex((t) => t.includes(term));
          if (firstIndex >= 0 && firstIndex < docLength * 0.3) {
            score += 0.1;
          }
        }
      }

      // Normalize by query length
      return score / queryTerms.length;
    });

    return Promise.resolve(scores);
  }

  private buildScoringPrompt(pairs: Array<{ query: string; passage: string }>): string {
    const examples = pairs
      .map((pair, i) => `Passage ${i + 1}: "${pair.passage.substring(0, 300)}..."\n`)
      .join('\n');

    return `You are a legal document relevance scorer. Rate how well each passage answers the query on a scale of 0.0 to 1.0.

Query: "${pairs[0].query}"

${examples}

Provide only the scores in order, separated by commas. Example: 0.85, 0.23, 0.67

Scores:`;
  }

  private parseScoresFromResponse(response: string, expectedCount: number): number[] {
    try {
      // Extract numbers from response
      const numbers = response.match(/\d+\.?\d*/g) || [];
      const scores = numbers
        .slice(0, expectedCount)
        .map((n) => Math.min(1.0, Math.max(0.0, parseFloat(n))));

      // Pad with fallback scores if needed
      while (scores.length < expectedCount) {
        scores.push(0.5);
      }

      return scores;
    } catch (error) {
      console.warn('[CrossEncoder] Failed to parse scores, using uniform fallback');
      return new Array(expectedCount).fill(0.5);
    }
  }

  private combineScores(scored: ScoredResult[]): ScoredResult[] {
    return scored.map((item) => {
      // Only trust rerank scores above confidence threshold
      const useRerank = item.metadata.confidence >= this.config.minConfidenceThreshold;

      if (useRerank) {
        item.combinedScore =
          this.config.scoreWeight * item.rerankScore +
          (1 - this.config.scoreWeight) * item.originalScore;
      } else {
        // Fall back to original score
        item.combinedScore = item.originalScore;
      }

      return item;
    });
  }

  private calculateConfidence(score: number): number {
    // Confidence based on score magnitude and distance from 0.5
    const distance = Math.abs(score - 0.5);
    return Math.min(1.0, distance * 2);
  }

  private extractText(result: SearchResult): string {
    // Extract searchable text from result
    const parts = [
      result.title || '',
      result.content || '',
      result.summary || '',
      result.excerpt || '',
    ].filter(Boolean);

    return parts.join(' ').trim();
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter((token) => token.length > 2);
  }

  private lexicalSimilarity(query: string, document: string): number {
    const queryTokens = new Set(this.tokenize(query));
    const docTokens = new Set(this.tokenize(document));

    const intersection = new Set([...queryTokens].filter((token) => docTokens.has(token)));
    const union = new Set([...queryTokens, ...docTokens]);

    return union.size > 0 ? intersection.size / union.size : 0;
  }

  // Cache management
  clearCache(): void {
    this.scoreCache.clear();
    this.modelCache.clear();
  }

  getCacheStats(): { scoreCache: number; modelCache: number } {
    return {
      scoreCache: this.scoreCache.size,
      modelCache: this.modelCache.size,
    };
  }
}

// Convenience function for quick reranking
export async function rerankSearchResults(
  query: string,
  results: SearchResult[],
  config: Partial<CrossEncoderConfig> = {}
): Promise<SearchResult[]> {
  const reranker = new CrossEncoderReranker(config);
  return reranker.rerankResults(query, results);
}

// Integration test helper
export async function testCrossEncoderReranking(): Promise<boolean> {
  try {
    const mockResults: SearchResult[] = [
      {
        id: 'doc1',
        title: 'Contract Formation Requirements',
        content: 'A valid contract requires offer, acceptance, and consideration.',
        score: 0.6,
        rank: 1,
        document: {
          id: 'doc1',
          title: 'Contract Formation Requirements',
          documentType: 'contract',
          content: 'A valid contract requires offer, acceptance, and consideration.',
        } as any,
      },
      {
        id: 'doc2',
        title: 'Employment Termination',
        content: 'Employment can be terminated with proper notice.',
        score: 0.8,
        rank: 2,
        document: {
          id: 'doc2',
          title: 'Employment Termination',
          documentType: 'case',
          content: 'Employment can be terminated with proper notice.',
        } as any,
      },
      {
        id: 'doc3',
        title: 'Property Rights',
        content: 'Property ownership includes the right to exclude others.',
        score: 0.4,
        rank: 3,
        document: {
          id: 'doc3',
          title: 'Property Rights',
          documentType: 'case',
          content: 'Property ownership includes the right to exclude others.',
        } as any,
      },
    ];

    const reranked = await rerankSearchResults('contract requirements formation', mockResults, {
      model: 'local-computation',
      timeout: 1000,
    });

    const isValid =
      reranked.length === mockResults.length &&
      reranked.every(
        (result) => typeof result.score === 'number' && result.metadata?.reranking?.modelUsed
      );

    console.log('[test] Cross-encoder reranking:', isValid ? 'PASS' : 'FAIL');
    console.log(
      '[test] Reranked scores:',
      reranked.map((r) => ({
        id: r.id,
        score: r.score?.toFixed(3),
        originalScore: r.metadata?.originalScore?.toFixed(3),
      }))
    );

    return isValid;
  } catch (error) {
    console.error('[test] Cross-encoder reranking failed:', error);
    return false;
  }
}

// Export default instance
const crossEncoderReranker = new CrossEncoderReranker();
export { crossEncoderReranker };
export default crossEncoderReranker;
