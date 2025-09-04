/**
 * GPU-Accelerated Semantic Embedding Service
 * Integrates nomic-embed-text with RAG service for high-performance embeddings
 */

import { uploadTelemetry } from './upload-telemetry-service';

export interface EmbeddingRequest {
  text: string | string[];
  model?: string;
  useGPU?: boolean;
  batchSize?: number;
  dimensions?: number;
}

export interface EmbeddingResponse {
  embeddings: number[][];
  model: string;
  dimensions: number;
  processingTime: number;
  gpuUsed: boolean;
  tokenCount: number;
}

export interface SemanticSearchRequest {
  query: string;
  documents: string[];
  threshold?: number;
  topK?: number;
  useGPU?: boolean;
}

export interface SemanticSearchResult {
  document: string;
  score: number;
  index: number;
  embedding?: number[];
}

export interface RAGContext {
  query: string;
  similarDocs: SemanticSearchResult[];
  embeddings: number[][];
  processingTime: number;
  metadata: {
    model: string;
    gpuUsed: boolean;
    vectorDimensions: number;
  };
}

class GPUSemanticEmbeddingService {
  private ollamaEndpoint = 'http://localhost:11434';
  private ragServiceEndpoint = 'http://localhost:8094';
  private defaultModel = 'nomic-embed-text:latest';
  private isInitialized = false;
  private gpuAvailable = false;
  
  constructor() {
    this.checkGPUAvailability();
  }

  /**
   * Check if GPU acceleration is available
   */
  private async checkGPUAvailability(): Promise<void> {
    try {
      const response = await fetch(`${this.ollamaEndpoint}/api/ps`);
      const models = await response.json();
      
      // Check if nomic-embed-text is loaded and GPU is available
      const nomicModel = models.models?.find((m: any) => 
        m.name?.includes('nomic-embed-text')
      );
      
      this.gpuAvailable = nomicModel?.details?.family?.includes('gpu') || false;
      this.isInitialized = true;
      
      console.log(`🔥 GPU Embedding Service: ${this.gpuAvailable ? 'GPU available' : 'CPU only'}`);
    } catch (error) {
      console.warn('Failed to check GPU availability:', error);
      this.gpuAvailable = false;
      this.isInitialized = true;
    }
  }

  /**
   * Generate embeddings for text or array of texts
   */
  async generateEmbeddings(request: EmbeddingRequest): Promise<EmbeddingResponse> {
    if (!this.isInitialized) {
      await this.checkGPUAvailability();
    }

    const startTime = Date.now();
    const texts = Array.isArray(request.text) ? request.text : [request.text];
    const model = request.model || this.defaultModel;
    const useGPU = request.useGPU !== false && this.gpuAvailable;
    
    uploadTelemetry.customEvent('embedding_start', {
      textCount: texts.length,
      model,
      useGPU,
      batchSize: request.batchSize || texts.length
    });

    try {
      // Process in batches for better GPU utilization
      const batchSize = request.batchSize || Math.min(texts.length, useGPU ? 32 : 16);
      const allEmbeddings: number[][] = [];
      
      for (let i = 0; i < texts.length; i += batchSize) {
        const batch = texts.slice(i, i + batchSize);
        const batchEmbeddings = await this.processBatch(batch, model, useGPU);
        allEmbeddings.push(...batchEmbeddings);
      }

      const processingTime = Date.now() - startTime;
      const tokenCount = this.estimateTokenCount(texts);

      uploadTelemetry.customEvent('embedding_complete', {
        textCount: texts.length,
        processingTime,
        model,
        gpuUsed: useGPU,
        tokenCount,
        dimensionsGenerated: allEmbeddings[0]?.length || 0
      });

      return {
        embeddings: allEmbeddings,
        model,
        dimensions: allEmbeddings[0]?.length || 384,
        processingTime,
        gpuUsed: useGPU,
        tokenCount
      };

    } catch (error) {
      uploadTelemetry.customEvent('embedding_error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        textCount: texts.length,
        model,
        useGPU
      });
      throw error;
    }
  }

  /**
   * Process a batch of texts for embedding
   */
  private async processBatch(texts: string[], model: string, useGPU: boolean): Promise<number[][]> {
    const embeddings: number[][] = [];
    
    for (const text of texts) {
      const response = await fetch(`${this.ollamaEndpoint}/api/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model,
          prompt: text,
          options: {
            num_gpu: useGPU ? 1 : 0,
            temperature: 0.0, // Deterministic embeddings
            num_ctx: 2048
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`Embedding generation failed: ${response.statusText}`);
      }

      const result = await response.json();
      if (result.embedding) {
        embeddings.push(result.embedding);
      } else {
        throw new Error('No embedding returned from Ollama');
      }
    }

    return embeddings;
  }

  /**
   * Perform semantic search using GPU-accelerated embeddings
   */
  async semanticSearch(request: SemanticSearchRequest): Promise<SemanticSearchResult[]> {
    const startTime = Date.now();
    
    try {
      // Generate embeddings for query and documents
      const [queryEmbedding, docEmbeddings] = await Promise.all([
        this.generateEmbeddings({ 
          text: request.query, 
          useGPU: request.useGPU 
        }),
        this.generateEmbeddings({ 
          text: request.documents, 
          useGPU: request.useGPU 
        })
      ]);

      // Calculate cosine similarities
      const queryVector = queryEmbedding.embeddings[0];
      const results: SemanticSearchResult[] = [];

      for (let i = 0; i < request.documents.length; i++) {
        const docVector = docEmbeddings.embeddings[i];
        const score = this.cosineSimilarity(queryVector, docVector);
        
        if (score >= (request.threshold || 0.3)) {
          results.push({
            document: request.documents[i],
            score,
            index: i,
            embedding: docVector
          });
        }
      }

      // Sort by similarity score and limit results
      results.sort((a, b) => b.score - a.score);
      const topResults = results.slice(0, request.topK || 10);

      const processingTime = Date.now() - startTime;
      
      uploadTelemetry.customEvent('semantic_search_complete', {
        queryLength: request.query.length,
        documentCount: request.documents.length,
        resultsFound: topResults.length,
        processingTime,
        gpuUsed: request.useGPU !== false && this.gpuAvailable
      });

      return topResults;

    } catch (error) {
      uploadTelemetry.customEvent('semantic_search_error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        queryLength: request.query.length,
        documentCount: request.documents.length
      });
      throw error;
    }
  }

  /**
   * Enhanced RAG query with GPU-accelerated embeddings
   */
  async enhancedRAGQuery(query: string, documents: string[], options: {
    useGPU?: boolean;
    model?: string;
    contextLimit?: number;
    temperature?: number;
  } = {}): Promise<RAGContext> {
    const startTime = Date.now();

    try {
      // Perform semantic search to find relevant documents
      const searchResults = await this.semanticSearch({
        query,
        documents,
        useGPU: options.useGPU,
        topK: options.contextLimit || 5,
        threshold: 0.4
      });

      // Generate embeddings for the final context
      const contextDocs = searchResults.map(r => r.document);
      const contextEmbeddings = await this.generateEmbeddings({
        text: contextDocs,
        model: options.model,
        useGPU: options.useGPU
      });

      // Call RAG service with enhanced context
      await this.callRAGService(query, searchResults, options);

      const processingTime = Date.now() - startTime;

      return {
        query,
        similarDocs: searchResults,
        embeddings: contextEmbeddings.embeddings,
        processingTime,
        metadata: {
          model: contextEmbeddings.model,
          gpuUsed: contextEmbeddings.gpuUsed,
          vectorDimensions: contextEmbeddings.dimensions
        }
      };

    } catch (error) {
      uploadTelemetry.customEvent('rag_query_error', {
        error: error instanceof Error ? error.message : 'Unknown error',
        query: query.substring(0, 100) + '...',
        documentCount: documents.length
      });
      throw error;
    }
  }

  /**
   * Call the RAG service with enhanced context
   */
  private async callRAGService(
    query: string, 
    context: SemanticSearchResult[], 
    options: any
  ): Promise<void> {
    try {
      const response = await fetch(`${this.ragServiceEndpoint}/api/rag`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query,
          context: context.map(r => ({
            content: r.document,
            score: r.score,
            embedding: r.embedding
          })),
          options: {
            model: options.model || 'gemma3-legal:latest',
            temperature: options.temperature || 0.7,
            useGPU: options.useGPU !== false && this.gpuAvailable
          }
        }),
      });

      if (!response.ok) {
        throw new Error(`RAG service error: ${response.statusText}`);
      }

    } catch (error) {
      console.error('RAG service call failed:', error);
      throw error;
    }
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) {
      throw new Error('Vector dimensions must match');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * Estimate token count for pricing/rate limiting
   */
  private estimateTokenCount(texts: string[]): number {
    return texts.reduce((total, text) => {
      // Rough estimation: ~0.75 tokens per word
      return total + Math.ceil(text.split(' ').length * 0.75);
    }, 0);
  }

  /**
   * Get service status and GPU information
   */
  async getStatus(): Promise<{
    initialized: boolean;
    gpuAvailable: boolean;
    ollamaConnected: boolean;
    ragServiceConnected: boolean;
    defaultModel: string;
  }> {
    const ollamaConnected = await this.checkServiceHealth(this.ollamaEndpoint);
    const ragServiceConnected = await this.checkServiceHealth(this.ragServiceEndpoint);

    return {
      initialized: this.isInitialized,
      gpuAvailable: this.gpuAvailable,
      ollamaConnected,
      ragServiceConnected,
      defaultModel: this.defaultModel
    };
  }

  /**
   * Check if a service endpoint is healthy
   */
  private async checkServiceHealth(endpoint: string): Promise<boolean> {
    try {
      const response = await fetch(`${endpoint}/api/health`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      return response.ok;
    } catch {
      return false;
    }
  }
}

// Export singleton instance
export const gpuEmbeddingService = new GPUSemanticEmbeddingService();

// Export types
export type {
  EmbeddingRequest,
  EmbeddingResponse,
  SemanticSearchRequest,
  SemanticSearchResult,
  RAGContext
};