/**
 * Phase 14 Evidence Processing - Unified Embedding Service
 * Resolves ONNX vs Ollama embedding conflicts
 * 
 * Strategy:
 * - Primary: ONNX for embeddings (faster, local)
 * - Fallback: Ollama for chat/generation
 * - Consistent: 384-dimensional embeddings
 */

import { OLLAMA_API_URL } from "$env/static/private";

// Type definitions
interface EmbeddingOptions {
  model?: 'onnx' | 'ollama';
  maxRetries?: number;
  timeout?: number;
}

interface EmbeddingResult {
  embedding: number[];
  dimensions: number;
  model: string;
  processingTime: number;
}

interface ChunkOptions {
  chunkSize?: number;
  chunkOverlap?: number;
  separator?: string;
}

interface TextChunk {
  content: string;
  startOffset: number;
  endOffset: number;
  tokens: number;
  chunkIndex: number;
}

class UnifiedEmbeddingService {
  private ollamaUrl: string;
  private embeddingModel: string;
  private chatModel: string;
  private dimensions: number;
  private onnxInitialized: boolean = false;

  constructor() {
    this.ollamaUrl = OLLAMA_API_URL || "http://localhost:11434";
    
    // PHASE 14 FIX: Separate models for different purposes
    this.embeddingModel = "nomic-embed-text"; // For embeddings (via ONNX)
    this.chatModel = "gemma3-legal:latest";   // For chat/generation (via Ollama)
    this.dimensions = 384; // Consistent embedding dimensions
  }

  /**
   * Initialize ONNX runtime for embeddings
   * Priority method for Phase 14 Evidence Processing
   */
  private async initializeONNX(): Promise<boolean> {
    try {
      // Dynamic import to avoid build-time issues
      const { pipeline } = await import('@xenova/transformers');
      
      // Initialize sentence transformer model
      this.onnxEmbedder = await pipeline(
        'feature-extraction',
        'sentence-transformers/all-MiniLM-L6-v2',
        { device: 'cpu' }
      );
      
      this.onnxInitialized = true;
      console.log('✅ ONNX embedding service initialized');
      return true;
    } catch (error) {
      console.warn('⚠️ ONNX initialization failed, falling back to Ollama:', error);
      this.onnxInitialized = false;
      return false;
    }
  }

  private onnxEmbedder: any = null;

  /**
   * Generate embeddings using ONNX (preferred method)
   */
  private async generateONNXEmbedding(text: string): Promise<number[]> {
    if (!this.onnxInitialized) {
      await this.initializeONNX();
    }

    if (!this.onnxEmbedder) {
      throw new Error('ONNX embedder not available');
    }

    const result = await this.onnxEmbedder(text, {
      pooling: 'mean',
      normalize: true
    });

    // Convert to flat array and ensure 384 dimensions
    const embedding = Array.from(result.data);
    
    // Pad or truncate to ensure consistent 384 dimensions
    if (embedding.length > this.dimensions) {
      return embedding.slice(0, this.dimensions);
    } else if (embedding.length < this.dimensions) {
      const padded = [...embedding];
      while (padded.length < this.dimensions) {
        padded.push(0);
      }
      return padded;
    }
    
    return embedding;
  }

  /**
   * Generate embeddings using Ollama (fallback method)
   */
  private async generateOllamaEmbedding(text: string): Promise<number[]> {
    const response = await fetch(`${this.ollamaUrl}/api/embeddings`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.embeddingModel,
        prompt: text
      })
    });

    if (!response.ok) {
      throw new Error(`Ollama embedding failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding || [];
  }

  /**
   * Unified embedding generation with automatic fallback
   * PHASE 14 EVIDENCE PROCESSING MAIN METHOD
   */
  async generateEmbedding(
    text: string, 
    options: EmbeddingOptions = {}
  ): Promise<EmbeddingResult> {
    const startTime = performance.now();
    const { model = 'onnx', maxRetries = 2, timeout = 30000 } = options;

    if (!text || text.trim().length === 0) {
      throw new Error('Text cannot be empty');
    }

    let embedding: number[] = [];
    let actualModel = model;
    let lastError: Error | null = null;

    // Try ONNX first (preferred for Phase 14)
    if (model === 'onnx' || model === 'ollama') {
      try {
        embedding = await this.generateONNXEmbedding(text);
        actualModel = 'onnx';
      } catch (error) {
        console.warn('ONNX embedding failed, trying Ollama fallback:', error);
        lastError = error as Error;
        
        try {
          embedding = await this.generateOllamaEmbedding(text);
          actualModel = 'ollama';
        } catch (ollamaError) {
          lastError = ollamaError as Error;
        }
      }
    }

    if (embedding.length === 0) {
      throw new Error(`All embedding methods failed. Last error: ${lastError?.message}`);
    }

    const processingTime = performance.now() - startTime;

    return {
      embedding,
      dimensions: this.dimensions,
      model: actualModel,
      processingTime
    };
  }

  /**
   * Generate embeddings for multiple texts (batch processing)
   * Optimized for Evidence Processing workloads
   */
  async generateBatchEmbeddings(
    texts: string[], 
    options: EmbeddingOptions = {}
  ): Promise<EmbeddingResult[]> {
    const batchSize = 10; // Process in chunks to avoid timeouts
    const results: EmbeddingResult[] = [];

    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchPromises = batch.map(text => this.generateEmbedding(text, options));
      
      try {
        const batchResults = await Promise.all(batchPromises);
        results.push(...batchResults);
      } catch (error) {
        console.error(`Batch embedding failed for batch ${i}-${i + batchSize}:`, error);
        throw error;
      }
    }

    return results;
  }

  /**
   * Text chunking for large documents
   * Essential for Evidence Processing
   */
  chunkText(text: string, options: ChunkOptions = {}): TextChunk[] {
    const {
      chunkSize = 512,
      chunkOverlap = 50,
      separator = '. '
    } = options;

    const sentences = text.split(separator);
    const chunks: TextChunk[] = [];
    let currentChunk = '';
    let startOffset = 0;
    let chunkIndex = 0;

    for (let i = 0; i < sentences.length; i++) {
      const sentence = sentences[i] + (i < sentences.length - 1 ? separator : '');
      
      if ((currentChunk + sentence).length > chunkSize && currentChunk.length > 0) {
        // Create chunk
        chunks.push({
          content: currentChunk.trim(),
          startOffset,
          endOffset: startOffset + currentChunk.length,
          tokens: Math.ceil(currentChunk.length / 4), // Rough token estimate
          chunkIndex
        });

        // Prepare next chunk with overlap
        const overlapText = currentChunk.slice(-chunkOverlap);
        currentChunk = overlapText + sentence;
        startOffset += currentChunk.length - overlapText.length;
        chunkIndex++;
      } else {
        currentChunk += sentence;
      }
    }

    // Add final chunk
    if (currentChunk.trim().length > 0) {
      chunks.push({
        content: currentChunk.trim(),
        startOffset,
        endOffset: startOffset + currentChunk.length,
        tokens: Math.ceil(currentChunk.length / 4),
        chunkIndex
      });
    }

    return chunks;
  }

  /**
   * Health check for embedding services
   */
  async getHealthStatus(): Promise<{
    onnx: boolean;
    ollama: boolean;
    overall: 'healthy' | 'degraded' | 'unhealthy';
  }> {
    const onnxHealthy = this.onnxInitialized || await this.initializeONNX();
    
    let ollamaHealthy = false;
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`, {
        method: 'GET',
        signal: AbortSignal.timeout(5000)
      });
      ollamaHealthy = response.ok;
    } catch {
      ollamaHealthy = false;
    }

    let overall: 'healthy' | 'degraded' | 'unhealthy';
    if (onnxHealthy && ollamaHealthy) {
      overall = 'healthy';
    } else if (onnxHealthy || ollamaHealthy) {
      overall = 'degraded';
    } else {
      overall = 'unhealthy';
    }

    return { onnx: onnxHealthy, ollama: ollamaHealthy, overall };
  }

  /**
   * Get chat completion using Ollama (separate from embeddings)
   * Phase 14: Clear separation of concerns
   */
  async getChatCompletion(
    messages: Array<{ role: string; content: string }>,
    options: { temperature?: number; maxTokens?: number } = {}
  ): Promise<string> {
    const { temperature = 0.7, maxTokens = 1000 } = options;

    const response = await fetch(`${this.ollamaUrl}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: this.chatModel, // Uses gemma3-legal:latest
        messages,
        stream: false,
        options: {
          temperature,
          num_predict: maxTokens
        }
      })
    });

    if (!response.ok) {
      throw new Error(`Chat completion failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.message?.content || '';
  }
}

// Export singleton instance
export const embeddingService = new UnifiedEmbeddingService();
export default embeddingService;

// Export types for use in other modules
export type { EmbeddingOptions, EmbeddingResult, ChunkOptions, TextChunk };