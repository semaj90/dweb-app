/**
 * Nomic Embedding Service
 * Advanced embedding generation service using nomic-embed-text
 * Features: CUDA acceleration, batch processing, caching, and semantic search
 */

import { OllamaEmbeddings } from '@langchain/ollama';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { Document } from '@langchain/core/documents';
import type { VectorStore } from '@langchain/core/vectorstores';

// Import our database service for persistent storage
import { db } from '$lib/database/connection';
import { searchIndex, type NewSearchIndex } from '$lib/database/schema';
import { eq, sql, desc, asc } from 'drizzle-orm';

export interface EmbeddingConfig {
  model: string;
  dimensions: number;
  batchSize: number;
  maxConcurrency: number;
  chunkSize: number;
  chunkOverlap: number;
  enableCaching: boolean;
  enableGpuAcceleration: boolean;
  normalization: boolean;
}

export interface DocumentChunk {
  id: string;
  content: string;
  metadata: {
    source: string;
    chunkIndex: number;
    totalChunks: number;
    startIndex: number;
    endIndex: number;
    [key: string]: any;
  };
}

export interface EmbeddingResult {
  id: string;
  embedding: number[];
  content: string;
  metadata: Record<string, any>;
  processingTime: number;
}

export interface SimilaritySearchResult {
  document: DocumentChunk;
  similarity: number;
  embedding: number[];
  metadata: Record<string, any>;
}

export interface BatchEmbeddingResult {
  results: EmbeddingResult[];
  totalProcessed: number;
  averageTime: number;
  errors: Array<{
    index: number;
    content: string;
    error: string;
  }>;
  metrics: {
    tokenCount: number;
    embeddingDimensions: number;
    cacheHits: number;
    cacheMisses: number;
  };
}

class NomicEmbeddingService {
  private static instance: NomicEmbeddingService;
  private embeddings: OllamaEmbeddings;
  private textSplitter: RecursiveCharacterTextSplitter;
  private vectorStore: VectorStore;
  private config: EmbeddingConfig;
  private cache: Map<string, { embedding: number[]; timestamp: number }> = new Map();
  private initialized = false;
  private processing = false;

  private constructor() {
    this.config = this.getDefaultConfig();
    this.initializeServices();
  }

  public static getInstance(): NomicEmbeddingService {
    if (!NomicEmbeddingService.instance) {
      NomicEmbeddingService.instance = new NomicEmbeddingService();
    }
    return NomicEmbeddingService.instance;
  }

  private getDefaultConfig(): EmbeddingConfig {
    return {
      model: 'nomic-embed-text:latest',
      dimensions: 768, // nomic-embed-text uses 768-dimensional embeddings
      batchSize: 32,   // Optimized for RTX 3060
      maxConcurrency: 4,
      chunkSize: 1000,  // Characters per chunk
      chunkOverlap: 200, // Overlap between chunks
      enableCaching: true,
      enableGpuAcceleration: true,
      normalization: true
    };
  }

  private async initializeServices(): Promise<void> {
    try {
      // Initialize Ollama embeddings with CUDA optimization
      this.embeddings = new OllamaEmbeddings({
        baseUrl: process.env.OLLAMA_BASE_URL || 'http://localhost:11434',
        model: this.config.model,
        requestOptions: {
          numGpu: this.config.enableGpuAcceleration ? 1 : 0,
          mainGpu: 0,
          batchSize: this.config.batchSize
        }
      });

      // Initialize text splitter for document chunking
      this.textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: this.config.chunkSize,
        chunkOverlap: this.config.chunkOverlap,
        separators: ['\n\n', '\n', '. ', ' ', '']
      });

      // Initialize in-memory vector store for fast similarity search
      this.vectorStore = new MemoryVectorStore(this.embeddings);

      this.initialized = true;
      console.log('✅ Nomic Embedding Service initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize Nomic Embedding Service:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings for a single text
   */
  public async generateEmbedding(
    text: string,
    metadata?: Record<string, any>
  ): Promise<EmbeddingResult> {
    const startTime = Date.now();
    
    try {
      if (!this.initialized) {
        await this.initializeServices();
      }

      // Check cache first
      const cacheKey = this.getCacheKey(text);
      if (this.config.enableCaching && this.cache.has(cacheKey)) {
        const cached = this.cache.get(cacheKey)!;
        return {
          id: this.generateId(),
          embedding: cached.embedding,
          content: text,
          metadata: metadata || {},
          processingTime: Date.now() - startTime
        };
      }

      // Generate embedding
      const embedding = await this.embeddings.embedQuery(text);
      
      // Normalize if enabled
      const normalizedEmbedding = this.config.normalization 
        ? this.normalizeVector(embedding)
        : embedding;

      // Cache result
      if (this.config.enableCaching) {
        this.cache.set(cacheKey, {
          embedding: normalizedEmbedding,
          timestamp: Date.now()
        });
      }

      const result: EmbeddingResult = {
        id: this.generateId(),
        embedding: normalizedEmbedding,
        content: text,
        metadata: metadata || {},
        processingTime: Date.now() - startTime
      };

      return result;
    } catch (error) {
      console.error('Failed to generate embedding:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings for multiple texts in batches
   */
  public async generateBatchEmbeddings(
    texts: string[],
    metadata?: Record<string, any>[],
    onProgress?: (processed: number, total: number) => void
  ): Promise<BatchEmbeddingResult> {
    const startTime = Date.now();
    this.processing = true;
    
    try {
      if (!this.initialized) {
        await this.initializeServices();
      }

      const results: EmbeddingResult[] = [];
      const errors: BatchEmbeddingResult['errors'] = [];
      let cacheHits = 0;
      let cacheMisses = 0;

      // Process in batches
      for (let i = 0; i < texts.length; i += this.config.batchSize) {
        const batch = texts.slice(i, i + this.config.batchSize);
        const batchMetadata = metadata?.slice(i, i + this.config.batchSize);

        try {
          // Check cache for batch items
          const cachedResults: EmbeddingResult[] = [];
          const uncachedTexts: string[] = [];
          const uncachedIndices: number[] = [];
          const uncachedMetadata: Record<string, any>[] = [];

          batch.forEach((text, batchIndex) => {
            const cacheKey = this.getCacheKey(text);
            if (this.config.enableCaching && this.cache.has(cacheKey)) {
              const cached = this.cache.get(cacheKey)!;
              cachedResults.push({
                id: this.generateId(),
                embedding: cached.embedding,
                content: text,
                metadata: batchMetadata?.[batchIndex] || {},
                processingTime: 0
              });
              cacheHits++;
            } else {
              uncachedTexts.push(text);
              uncachedIndices.push(batchIndex);
              uncachedMetadata.push(batchMetadata?.[batchIndex] || {});
              cacheMisses++;
            }
          });

          // Process uncached texts
          if (uncachedTexts.length > 0) {
            const embeddings = await this.embeddings.embedDocuments(uncachedTexts);
            
            embeddings.forEach((embedding, index) => {
              const text = uncachedTexts[index];
              const normalizedEmbedding = this.config.normalization 
                ? this.normalizeVector(embedding)
                : embedding;

              // Cache result
              if (this.config.enableCaching) {
                this.cache.set(this.getCacheKey(text), {
                  embedding: normalizedEmbedding,
                  timestamp: Date.now()
                });
              }

              cachedResults.push({
                id: this.generateId(),
                embedding: normalizedEmbedding,
                content: text,
                metadata: uncachedMetadata[index],
                processingTime: (Date.now() - startTime) / uncachedTexts.length
              });
            });
          }

          results.push(...cachedResults);
        } catch (error) {
          // Handle batch errors
          batch.forEach((text, batchIndex) => {
            errors.push({
              index: i + batchIndex,
              content: text.substring(0, 100) + '...',
              error: error instanceof Error ? error.message : 'Unknown error'
            });
          });
        }

        // Report progress
        if (onProgress) {
          onProgress(Math.min(i + this.config.batchSize, texts.length), texts.length);
        }

        // Add small delay to prevent overwhelming the service
        if (i + this.config.batchSize < texts.length) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }

      const totalTime = Date.now() - startTime;
      const averageTime = results.length > 0 ? totalTime / results.length : 0;

      this.processing = false;

      return {
        results,
        totalProcessed: results.length,
        averageTime,
        errors,
        metrics: {
          tokenCount: this.estimateTokenCount(texts),
          embeddingDimensions: this.config.dimensions,
          cacheHits,
          cacheMisses
        }
      };
    } catch (error) {
      this.processing = false;
      console.error('Failed to generate batch embeddings:', error);
      throw error;
    }
  }

  /**
   * Process and embed a document with automatic chunking
   */
  public async processDocument(
    content: string,
    metadata: {
      source: string;
      title?: string;
      entityType: string;
      entityId: string;
      [key: string]: any;
    }
  ): Promise<{
    chunks: DocumentChunk[];
    embeddings: EmbeddingResult[];
    indexedCount: number;
  }> {
    try {
      if (!this.initialized) {
        await this.initializeServices();
      }

      // Split document into chunks
      const textChunks = await this.textSplitter.splitText(content);
      
      // Create document chunks with metadata
      const chunks: DocumentChunk[] = textChunks.map((chunk, index) => ({
        id: this.generateId(),
        content: chunk,
        metadata: {
          ...metadata,
          chunkIndex: index,
          totalChunks: textChunks.length,
          startIndex: content.indexOf(chunk),
          endIndex: content.indexOf(chunk) + chunk.length
        }
      }));

      // Generate embeddings for all chunks
      const chunkTexts = chunks.map(chunk => chunk.content);
      const chunkMetadata = chunks.map(chunk => chunk.metadata);
      
      const embeddingResult = await this.generateBatchEmbeddings(
        chunkTexts, 
        chunkMetadata,
        (processed, total) => {
          console.log(`Processing document chunks: ${processed}/${total}`);
        }
      );

      // Store embeddings in database
      const indexedCount = await this.storeEmbeddingsInDatabase(
        embeddingResult.results,
        metadata.entityType,
        metadata.entityId
      );

      console.log(`✅ Processed document: ${chunks.length} chunks, ${indexedCount} indexed`);

      return {
        chunks,
        embeddings: embeddingResult.results,
        indexedCount
      };
    } catch (error) {
      console.error('Failed to process document:', error);
      throw error;
    }
  }

  /**
   * Perform semantic similarity search
   */
  public async similaritySearch(
    query: string,
    options: {
      k?: number;
      threshold?: number;
      entityType?: string;
      entityId?: string;
      metadata?: Record<string, any>;
    } = {}
  ): Promise<SimilaritySearchResult[]> {
    try {
      const {
        k = 10,
        threshold = 0.7,
        entityType,
        entityId,
        metadata
      } = options;

      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);

      // Search in database
      let searchQuery = db
        .select()
        .from(searchIndex)
        .limit(k * 2); // Get more results to filter

      if (entityType) {
        searchQuery = searchQuery.where(eq(searchIndex.entityType, entityType));
      }

      if (entityId) {
        searchQuery = searchQuery.where(eq(searchIndex.entityId, entityId));
      }

      const results = await searchQuery;

      // Calculate similarities and filter
      const similarities: SimilaritySearchResult[] = [];

      for (const result of results) {
        if (result.embedding) {
          const similarity = this.cosineSimilarity(
            queryEmbedding.embedding,
            result.embedding as number[]
          );

          if (similarity >= threshold) {
            similarities.push({
              document: {
                id: result.id,
                content: result.content,
                metadata: {
                  ...result.metadata as Record<string, any>,
                  entityType: result.entityType,
                  entityId: result.entityId
                }
              } as DocumentChunk,
              similarity,
              embedding: result.embedding as number[],
              metadata: result.metadata as Record<string, any>
            });
          }
        }
      }

      // Sort by similarity and return top k
      return similarities
        .sort((a, b) => b.similarity - a.similarity)
        .slice(0, k);
    } catch (error) {
      console.error('Failed to perform similarity search:', error);
      throw error;
    }
  }

  /**
   * Store embeddings in PostgreSQL with pgvector
   */
  private async storeEmbeddingsInDatabase(
    embeddings: EmbeddingResult[],
    entityType: string,
    entityId: string
  ): Promise<number> {
    try {
      let insertedCount = 0;

      for (const embedding of embeddings) {
        const indexEntry: NewSearchIndex = {
          entityType,
          entityId,
          content: embedding.content,
          embedding: embedding.embedding,
          metadata: embedding.metadata
        };

        await db.insert(searchIndex).values(indexEntry);
        insertedCount++;
      }

      return insertedCount;
    } catch (error) {
      console.error('Failed to store embeddings in database:', error);
      return 0;
    }
  }

  /**
   * Get embedding service statistics
   */
  public getStatistics(): {
    cacheSize: number;
    cacheHitRate: number;
    isProcessing: boolean;
    config: EmbeddingConfig;
    performance: {
      averageTime: number;
      totalProcessed: number;
    };
  } {
    return {
      cacheSize: this.cache.size,
      cacheHitRate: 0, // Would need to track this over time
      isProcessing: this.processing,
      config: this.config,
      performance: {
        averageTime: 0, // Would need to track this over time
        totalProcessed: 0 // Would need to track this over time
      }
    };
  }

  /**
   * Clear embedding cache
   */
  public clearCache(): void {
    this.cache.clear();
    console.log('✅ Embedding cache cleared');
  }

  /**
   * Update service configuration
   */
  public async updateConfig(newConfig: Partial<EmbeddingConfig>): Promise<void> {
    this.config = { ...this.config, ...newConfig };
    
    // Reinitialize if necessary
    if (newConfig.model || newConfig.enableGpuAcceleration) {
      await this.initializeServices();
    }
    
    console.log('✅ Embedding service configuration updated');
  }

  // Utility methods
  private getCacheKey(text: string): string {
    // Simple hash function for cache key
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString();
  }

  private generateId(): string {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  private normalizeVector(vector: number[]): number[] {
    const magnitude = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return magnitude > 0 ? vector.map(val => val / magnitude) : vector;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;
    
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    const magnitude = Math.sqrt(normA) * Math.sqrt(normB);
    return magnitude > 0 ? dotProduct / magnitude : 0;
  }

  private estimateTokenCount(texts: string[]): number {
    return texts.reduce((total, text) => total + Math.ceil(text.length / 4), 0);
  }

  // Getters
  public get isInitialized(): boolean {
    return this.initialized;
  }

  public get isProcessing(): boolean {
    return this.processing;
  }

  public get currentConfig(): EmbeddingConfig {
    return { ...this.config };
  }
}

// Export singleton instance
export const nomicEmbeddingService = NomicEmbeddingService.getInstance();
export default nomicEmbeddingService;

// Export types
export type {
  EmbeddingConfig,
  DocumentChunk,
  EmbeddingResult,
  SimilaritySearchResult,
  BatchEmbeddingResult
};