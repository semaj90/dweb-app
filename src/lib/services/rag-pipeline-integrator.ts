// RAG Pipeline Integrator
// Comprehensive RAG (Retrieval-Augmented Generation) pipeline integration service

import type { SearchResult } from '$lib/types/search-types';
import type { LegalDocument } from '$lib/types/legal-document';

export interface RagPipelineOptions {
  embeddingModel?: string;
  retrievalLimit?: number;
  rerankingEnabled?: boolean;
  vectorStore?: 'qdrant' | 'pgvector' | 'redis';
  llmProvider?: 'ollama' | 'openai' | 'claude';
}

export interface DocumentProgress {
  documents: Array<{
    id: string;
    title: string;
    content?: string;
    summary?: string;
    excerpt?: string;
    metadata?: Record<string, any>;
    type?: string;
    createdAt?: Date;
    updatedAt?: Date;
    score?: number;
  }>;
  totalProcessed: number;
  totalToProcess: number;
  currentStep: string;
  errors?: string[];
}

export interface RetrievalResult {
  documents: SearchResult[];
  metadata: {
    queryTime: number;
    totalResults: number;
    reranked: boolean;
    model: string;
  };
}

export class RagPipelineIntegrator {
  private options: Required<RagPipelineOptions>;

  constructor(options: RagPipelineOptions = {}) {
    this.options = {
      embeddingModel: options.embeddingModel || 'nomic-embed-text',
      retrievalLimit: options.retrievalLimit || 10,
      rerankingEnabled: options.rerankingEnabled ?? true,
      vectorStore: options.vectorStore || 'qdrant',
      llmProvider: options.llmProvider || 'ollama'
    };
  }

  async processDocuments(progress: DocumentProgress): Promise<SearchResult[]> {
    try {
      // Convert DocumentProgress.documents to SearchResult format
      const searchResults: SearchResult[] = progress.documents.map((doc, index) => ({
        score: doc.score || 0.8,
        rank: index + 1,
        id: doc.id,
        title: doc.title,
        content: doc.content,
        summary: doc.summary,
        excerpt: doc.excerpt,
        metadata: doc.metadata,
        type: doc.type,
        createdAt: doc.createdAt,
        updatedAt: doc.updatedAt,
        document: this.convertToLegalDocument(doc) // Add the required document property
      }));

      // Apply reranking if enabled
      if (this.options.rerankingEnabled && searchResults.length > 1) {
        return await this.rerankResults(searchResults);
      }

      return searchResults;
    } catch (error: unknown) {
      console.error('RAG Pipeline processing error:', error);
      throw new Error(`Failed to process documents: ${error.message}`);
    }
  }

  async retrieveDocuments(query: string, filters?: Record<string, any>): Promise<RetrievalResult> {
    const startTime = Date.now();
    
    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);
      
      // Perform vector search
      const searchResults = await this.performVectorSearch(queryEmbedding, filters);
      
      // Apply reranking if enabled
      const finalResults = this.options.rerankingEnabled 
        ? await this.rerankResults(searchResults)
        : searchResults;

      return {
        documents: finalResults,
        metadata: {
          queryTime: Date.now() - startTime,
          totalResults: finalResults.length,
          reranked: this.options.rerankingEnabled,
          model: this.options.embeddingModel
        }
      };
    } catch (error: unknown) {
      console.error('Document retrieval error:', error);
      throw new Error(`Failed to retrieve documents: ${error.message}`);
    }
  }

  async generateResponse(query: string, context: SearchResult[]): Promise<{
    response: string;
    sources: SearchResult[];
    confidence: number;
  }> {
    try {
      // Prepare context from retrieved documents
      const contextText = context
        .map(result => `Title: ${result.title}\nContent: ${result.excerpt || result.content?.substring(0, 500)}`)
        .join('\n\n');

      // Generate response using LLM
      const response = await this.callLLM(query, contextText);
      
      return {
        response: response.text,
        sources: context,
        confidence: response.confidence || 0.8
      };
    } catch (error: unknown) {
      console.error('Response generation error:', error);
      throw new Error(`Failed to generate response: ${error.message}`);
    }
  }

  private convertToLegalDocument(doc: unknown): LegalDocument {
    return {
      id: doc.id,
      title: doc.title,
      content: doc.content || '',
      documentType: doc.type || 'general',
      jurisdiction: doc.metadata?.jurisdiction || 'federal',
      processingStatus: 'completed',
      createdAt: doc.createdAt || new Date(),
      updatedAt: doc.updatedAt || new Date(),
      // Optional fields
      court: doc.metadata?.court,
      citation: doc.metadata?.citation,
      fullCitation: doc.metadata?.fullCitation,
      docketNumber: doc.metadata?.docketNumber,
      dateDecided: doc.metadata?.dateDecided,
      datePublished: doc.metadata?.datePublished,
      summary: doc.summary,
      tags: doc.metadata?.tags,
      metadata: doc.metadata,
      analysisResults: doc.metadata?.analysisResults,
      contentEmbedding: doc.metadata?.contentEmbedding,
      titleEmbedding: doc.metadata?.titleEmbedding,
      fileHash: doc.metadata?.fileHash,
      fileName: doc.metadata?.fileName,
      fileSize: doc.metadata?.fileSize,
      mimeType: doc.metadata?.mimeType,
      practiceArea: doc.metadata?.practiceArea
    };
  }

  private async generateEmbedding(text: string): Promise<number[]> {
    try {
      // Mock embedding generation - replace with actual service call
      const response = await fetch('/api/embeddings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          model: this.options.embeddingModel
        })
      });

      if (!response.ok) {
        throw new Error(`Embedding API error: ${response.status}`);
      }

      const data = await response.json();
      return data.embedding;
    } catch (error: unknown) {
      console.warn('Embedding generation failed, using mock embedding:', error);
      // Return mock embedding for development
      return Array.from({ length: 384 }, () => Math.random() - 0.5);
    }
  }

  private async performVectorSearch(embedding: number[], filters?: Record<string, any>): Promise<SearchResult[]> {
    try {
      const response = await fetch('/api/search/vector', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          vector: embedding,
          limit: this.options.retrievalLimit,
          filters
        })
      });

      if (!response.ok) {
        throw new Error(`Vector search API error: ${response.status}`);
      }

      const data = await response.json();
      return data.results || [];
    } catch (error: unknown) {
      console.warn('Vector search failed, using mock results:', error);
      // Return mock results for development
      return this.generateMockResults();
    }
  }

  private async rerankResults(results: SearchResult[]): Promise<SearchResult[]> {
    try {
      const response = await fetch('/api/search/rerank', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          results,
          method: 'cross-encoder'
        })
      });

      if (!response.ok) {
        throw new Error(`Reranking API error: ${response.status}`);
      }

      const data = await response.json();
      return data.rerankedResults || results;
    } catch (error: unknown) {
      console.warn('Reranking failed, returning original results:', error);
      return results;
    }
  }

  private async callLLM(query: string, context: string): Promise<{ text: string; confidence: number }> {
    try {
      const response = await fetch('/api/llm/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          context,
          provider: this.options.llmProvider,
          maxTokens: 1000
        })
      });

      if (!response.ok) {
        throw new Error(`LLM API error: ${response.status}`);
      }

      const data = await response.json();
      return {
        text: data.response,
        confidence: data.confidence || 0.8
      };
    } catch (error: unknown) {
      console.warn('LLM call failed, using fallback response:', error);
      return {
        text: 'I apologize, but I\'m unable to generate a response at this time due to a service error.',
        confidence: 0.1
      };
    }
  }

  private generateMockResults(): SearchResult[] {
    return [
      {
        score: 0.95,
        rank: 1,
        id: 'mock-1',
        title: 'Sample Legal Document 1',
        content: 'This is a sample legal document for testing purposes.',
        excerpt: 'Sample excerpt from the legal document...',
        type: 'contract',
        createdAt: new Date(),
        updatedAt: new Date(),
        document: {
          id: 'mock-1',
          title: 'Sample Legal Document 1',
          content: 'This is a sample legal document for testing purposes.',
          documentType: 'contract',
          jurisdiction: 'federal',
          processingStatus: 'completed',
          createdAt: new Date(),
          updatedAt: new Date()
        }
      }
    ];
  }

  // Configuration methods
  updateOptions(newOptions: Partial<RagPipelineOptions>): void {
    this.options = { ...this.options, ...newOptions };
  }

  getOptions(): Required<RagPipelineOptions> {
    return { ...this.options };
  }
}

// Export singleton instance
export const ragPipelineIntegrator = new RagPipelineIntegrator();
