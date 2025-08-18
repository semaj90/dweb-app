/**
 * RAG Pipeline Integration Service
 * Orchestrates enhanced text processing, MMR summarization, and cross-encoder reranking
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

interface SummaryRequest {
  documents: LegalDocument[];
  query?: string;
  maxLength?: number;
  diversityLambda?: number;
}

interface RAGPipelineConfig {
  enableSentenceSplitting: boolean;
  enableMMRSummarization: boolean;
  enableCrossEncoderReranking: boolean;
  maxDocuments: number;
  maxSummaryLength: number;
  rerankThreshold: number;
  cacheResults: boolean;
  enableStreaming: boolean;
}

interface RAGPipelineResult {
  query: string;
  documents: LegalDocument[];
  rerankedResults: SearchResult[];
  summary: string;
  metadata: {
    processingTime: number;
    documentsProcessed: number;
    sentencesExtracted: number;
    summaryGenerated: boolean;
    rerankingApplied: boolean;
    cacheHit: boolean;
  };
  confidence: number;
}

export class RAGPipelineIntegrator {
  private config: RAGPipelineConfig;
  private resultCache: Map<string, RAGPipelineResult> = new Map();

  constructor(config: Partial<RAGPipelineConfig> = {}) {
    this.config = {
      enableSentenceSplitting: true,
      enableMMRSummarization: true,
      enableCrossEncoderReranking: true,
      maxDocuments: 20,
      maxSummaryLength: 500,
      rerankThreshold: 0.5,
      cacheResults: true,
      enableStreaming: true,
      ...config,
    };
  }

  async processRAGQuery(
    query: string,
    documents: LegalDocument[],
    request?: Partial<SummaryRequest>
  ): Promise<RAGPipelineResult> {
    const startTime = Date.now();
    const cacheKey = this.generateCacheKey(query, documents);

    // Check cache first
    if (this.config.cacheResults && this.resultCache.has(cacheKey)) {
      const cached = this.resultCache.get(cacheKey)!;
      cached.metadata.cacheHit = true;
      return cached;
    }

    try {
      // Step 1: Enhanced sentence splitting for better text processing
      let processedDocuments = documents;
      let sentencesExtracted = 0;

      if (this.config.enableSentenceSplitting) {
        processedDocuments = await this.applySentenceSplitting(documents);
        sentencesExtracted = this.countSentences(processedDocuments);
      }

      // Step 2: Initial document ranking and filtering
      const filteredDocuments = processedDocuments.slice(0, this.config.maxDocuments);

      // Convert to SearchResult format for reranking
      const searchResults: SearchResult[] = filteredDocuments.map((doc, index) => ({
        document: doc,
        score: this.calculateInitialScore(doc, query),
        rank: index + 1,
        id: doc.id,
        title: doc.title,
        content: doc.content,
        summary: doc.summary,
        excerpt: doc.excerpt,
        metadata: doc.metadata
      }));

      // Step 3: Cross-encoder reranking for improved relevance
      let rerankedResults = searchResults;
      let rerankingApplied = false;

      if (this.config.enableCrossEncoderReranking) {
        rerankedResults = await this.applyCrossEncoderReranking(query, searchResults);
        rerankingApplied = true;
      }

      // Step 4: MMR-based summarization with diversity optimization
      let summary = '';
      let summaryGenerated = false;

      if (this.config.enableMMRSummarization) {
        const topDocuments = rerankedResults
          .filter((result) => result.score >= this.config.rerankThreshold)
          .slice(0, 10)
          .map((result) => result as LegalDocument);

        if (topDocuments.length > 0) {
          const summaryResult = await this.generateMMRSummary(query, topDocuments, request);
          summary = summaryResult.summary;
          summaryGenerated = true;
        }
      }

      // Fallback summary if MMR fails
      if (!summaryGenerated && rerankedResults.length > 0) {
        summary = this.generateFallbackSummary(rerankedResults, query);
      }

      const result: RAGPipelineResult = {
        query,
        documents: filteredDocuments,
        rerankedResults,
        summary,
        metadata: {
          processingTime: Date.now() - startTime,
          documentsProcessed: filteredDocuments.length,
          sentencesExtracted,
          summaryGenerated,
          rerankingApplied,
          cacheHit: false,
        },
        confidence: this.calculateOverallConfidence(rerankedResults, summaryGenerated),
      };

      // Cache result
      if (this.config.cacheResults) {
        this.resultCache.set(cacheKey, result);
      }

      return result;
    } catch (error) {
      console.error('[RAGPipeline] Processing failed:', error);

      // Return minimal fallback result
      return {
        query,
        documents: documents.slice(0, 5),
        rerankedResults: documents.slice(0, 5).map(
          (doc, i) =>
            ({
              document: doc,
              score: 0.5,
              rank: i + 1,
              id: doc.id,
              title: doc.title,
              content: doc.content,
              summary: doc.summary,
              excerpt: doc.excerpt,
              metadata: doc.metadata
            }) as SearchResult
        ),
        summary: `Analysis of "${query}" resulted in ${documents.length} relevant documents.`,
        metadata: {
          processingTime: Date.now() - startTime,
          documentsProcessed: documents.length,
          sentencesExtracted: 0,
          summaryGenerated: false,
          rerankingApplied: false,
          cacheHit: false,
        },
        confidence: 0.3,
      };
    }
  }

  async processRAGQueryStreaming(
    query: string,
    documents: LegalDocument[],
    request?: Partial<SummaryRequest>
  ): Promise<ReadableStream<string>> {
    if (!this.config.enableStreaming) {
      // Non-streaming fallback
      const result = await this.processRAGQuery(query, documents, request);
      return new ReadableStream({
        start(controller) {
          controller.enqueue(JSON.stringify({ type: 'complete', data: result }));
          controller.close();
        },
      });
    }

    return new ReadableStream({
      async start(controller) {
        try {
          const startTime = Date.now();

          // Initialize progress tracking
          let progress = {
            stage: 'initializing',
            progress: 0,
            documents: documents.slice(0, this.config.maxDocuments),
            rerankedResults: [],
            summary: '',
            metadata: {
              processingTime: 0,
              documentsProcessed: 0,
              sentencesExtracted: 0,
              summaryGenerated: false,
              rerankingApplied: false,
              cacheHit: false,
            },
          };

          // Send initial state
          controller.enqueue(
            JSON.stringify({
              type: 'progress',
              data: progress,
              timestamp: Date.now(),
            }) + '\n'
          );

          // Step 1: Sentence splitting
          if (this.config.enableSentenceSplitting) {
            progress.stage = 'sentence_splitting';
            progress.progress = 20;
            controller.enqueue(
              JSON.stringify({
                type: 'progress',
                data: progress,
                timestamp: Date.now(),
              }) + '\n'
            );

            progress.documents = await this.applySentenceSplitting(documents);
            progress.metadata.sentencesExtracted = this.countSentences(progress.documents);
          }

          // Step 2: Initial ranking
          progress.stage = 'initial_ranking';
          progress.progress = 40;
          controller.enqueue(
            JSON.stringify({
              type: 'progress',
              data: progress,
              timestamp: Date.now(),
            }) + '\n'
          );

          const searchResults: SearchResult[] = progress.documents.map((doc, index) => ({
            document: doc,
            score: this.calculateInitialScore(doc, query),
            rank: index + 1,
            id: doc.id,
            title: doc.title,
            content: doc.content,
            summary: doc.summary,
            excerpt: doc.excerpt,
            metadata: doc.metadata
          }));

          // Step 3: Cross-encoder reranking
          if (this.config.enableCrossEncoderReranking) {
            progress.stage = 'reranking';
            progress.progress = 60;
            controller.enqueue(
              JSON.stringify({
                type: 'progress',
                data: progress,
                timestamp: Date.now(),
              }) + '\n'
            );

            progress.rerankedResults = await this.applyCrossEncoderReranking(query, searchResults);
            progress.metadata.rerankingApplied = true;
          } else {
            progress.rerankedResults = searchResults;
          }

          // Step 4: MMR summarization
          if (this.config.enableMMRSummarization) {
            progress.stage = 'summarization';
            progress.progress = 80;
            controller.enqueue(
              JSON.stringify({
                type: 'progress',
                data: progress,
                timestamp: Date.now(),
              }) + '\n'
            );

            const topDocs = progress.rerankedResults
              .filter((r) => r.score >= this.config.rerankThreshold)
              .slice(0, 10)
              .map((r) => r as LegalDocument);

            if (topDocs.length > 0) {
              const summaryResult = await this.generateMMRSummary(query, topDocs, request);
              progress.summary = summaryResult.summary;
              progress.metadata.summaryGenerated = true;
            }
          }

          // Final result
          progress.stage = 'completed';
          progress.progress = 100;
          progress.metadata.processingTime = Date.now() - startTime;
          progress.metadata.documentsProcessed = progress.documents.length;

          controller.enqueue(
            JSON.stringify({
              type: 'complete',
              data: progress,
              timestamp: Date.now(),
            }) + '\n'
          );

          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });
  }

  private async applySentenceSplitting(documents: LegalDocument[]): Promise<LegalDocument[]> {
    try {
  const { splitSentencesEnhanced } = await import('$text/enhanced-sentence-splitter');

      return documents.map((doc) => ({
        ...doc,
        content: splitSentencesEnhanced(doc.content).join(' '),
        metadata: {
          ...doc.metadata,
          sentenceSplittingApplied: true,
          originalLength: doc.content.length,
        },
      }));
    } catch (error) {
      console.warn('[RAGPipeline] Sentence splitting failed, using original documents:', error);
      return documents;
    }
  }

  private async applyCrossEncoderReranking(
    query: string,
    searchResults: SearchResult[]
  ): Promise<SearchResult[]> {
    try {
      const { rerankSearchResults } = await import('./cross-encoder-reranker');
      return await rerankSearchResults(query, searchResults, {
        maxResults: this.config.maxDocuments,
        timeout: 5000,
      });
    } catch (error) {
      console.warn('[RAGPipeline] Cross-encoder reranking failed, using original ranking:', error);
      return searchResults;
    }
  }

  private async generateMMRSummary(
    query: string,
    documents: LegalDocument[],
    request?: Partial<SummaryRequest>
  ): Promise<{ summary: string }> {
    try {
      const { generateMMRSummary } = await import('./mmr-summary-generator');
      const config = {
        maxSummaryLength: request?.maxLength || this.config.maxSummaryLength,
        maxSentences: 5,
        lambda: 0.7,
      };
      const result = await generateMMRSummary(documents, query, config);
      return { summary: result.summary };
    } catch (error) {
      console.warn('[RAGPipeline] MMR summarization failed, using fallback:', error);
      return { summary: this.generateFallbackSummary(documents as SearchResult[], query) };
    }
  }

  private calculateInitialScore(document: LegalDocument, query: string): number {
    const queryTerms = query.toLowerCase().split(/\s+/);
    const docText = (document.content + ' ' + (document.title || '')).toLowerCase();

    let score = 0;
    const totalTerms = queryTerms.length;

    for (const term of queryTerms) {
      if (docText.includes(term)) {
        // Title match gets higher weight
        if (document.title?.toLowerCase().includes(term)) {
          score += 0.3;
        }
        // Content match
        const termCount = (docText.match(new RegExp(term, 'g')) || []).length;
        score += Math.min(0.2 * termCount, 0.5);
      }
    }

    return Math.min(score / totalTerms, 1.0);
  }

  private countSentences(documents: LegalDocument[]): number {
    return documents.reduce((total, doc) => {
      return total + (doc.content.match(/[.!?]+/g) || []).length;
    }, 0);
  }

  private generateFallbackSummary(results: SearchResult[], query: string): string {
    if (results.length === 0) {
      return `No relevant documents found for query: "${query}"`;
    }

    const topResult = results[0];
    const excerpt = topResult.content.substring(0, this.config.maxSummaryLength - 100);

    return `Based on ${results.length} documents, key findings for "${query}": ${excerpt}...`;
  }

  private calculateOverallConfidence(results: SearchResult[], summaryGenerated: boolean): number {
    if (results.length === 0) return 0;

    const avgScore = results.reduce((sum, r) => sum + r.score, 0) / results.length;
    const resultCountFactor = Math.min(results.length / 5, 1.0);
    const summaryFactor = summaryGenerated ? 1.0 : 0.8;

    return avgScore * resultCountFactor * summaryFactor;
  }

  private generateCacheKey(query: string, documents: LegalDocument[]): string {
    const docIds = documents
      .slice(0, 10)
      .map((d) => d.id)
      .join(',');
    return `${query.toLowerCase().trim()}_${docIds}`;
  }

  // Cache management
  clearCache(): void {
    this.resultCache.clear();
  }

  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.resultCache.size,
      keys: Array.from(this.resultCache.keys()).slice(0, 10),
    };
  }
}

// Convenience functions
export async function processLegalQuery(
  query: string,
  documents: LegalDocument[],
  config?: Partial<RAGPipelineConfig>
): Promise<RAGPipelineResult> {
  const pipeline = new RAGPipelineIntegrator(config);
  return pipeline.processRAGQuery(query, documents);
}

export async function processLegalQueryStreaming(
  query: string,
  documents: LegalDocument[],
  config?: Partial<RAGPipelineConfig>
): Promise<ReadableStream<string>> {
  const pipeline = new RAGPipelineIntegrator(config);
  return pipeline.processRAGQueryStreaming(query, documents);
}

// Integration test helper
export async function testRAGPipelineIntegration(): Promise<boolean> {
  try {
    const mockDocuments: LegalDocument[] = [
      {
        id: 'doc1',
        title: 'Contract Formation Law',
        content:
          'A valid contract requires offer, acceptance, and consideration. The parties must have legal capacity.',
        type: 'legal',
        createdAt: new Date(),
        updatedAt: new Date(),
      },
      {
        id: 'doc2',
        title: 'Employment Termination',
        content:
          'Employment contracts may be terminated with proper notice. Wrongful termination claims require proof.',
        type: 'legal',
        createdAt: new Date(),
        updatedAt: new Date(),
      },
    ];

    const result = await processLegalQuery('contract formation requirements', mockDocuments, {
      maxDocuments: 5,
      maxSummaryLength: 200,
    });

    const isValid =
      result.summary.length > 0 &&
      result.rerankedResults.length > 0 &&
      result.confidence > 0 &&
      result.metadata.processingTime > 0;

    console.log('[test] RAG pipeline integration:', isValid ? 'PASS' : 'FAIL');
    console.log('[test] Result summary:', result.summary.substring(0, 100) + '...');
    console.log('[test] Processed documents:', result.metadata.documentsProcessed);
    console.log('[test] Confidence:', result.confidence.toFixed(3));

    return isValid;
  } catch (error) {
    console.error('[test] RAG pipeline integration failed:', error);
    return false;
  }
}
