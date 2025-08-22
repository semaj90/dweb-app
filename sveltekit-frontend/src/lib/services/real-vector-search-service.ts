import { QdrantClient } from '@qdrant/js-client-rest';

/**
 * Real Vector Search Service - No Mocks
 * Integrates with Ollama (embeddings) + Qdrant (vector storage) + PostgreSQL (metadata)
 */

export interface VectorSearchOptions {
  maxResults?: number;
  threshold?: number;
  collection?: string;
  includeMetadata?: boolean;
  filter?: Record<string, any>;
}

export interface VectorSearchResult {
  id: string;
  content: string;
  score: number;
  metadata?: Record<string, any>;
}

export interface SearchResponse {
  success: boolean;
  results: VectorSearchResult[];
  totalResults: number;
  queryTime: number;
  model: string;
}

export class RealVectorSearchService {
  private qdrantClient: QdrantClient;
  private ollamaBaseUrl: string;
  private embeddingModel: string;

  constructor(options?: {
    qdrantUrl?: string;
    ollamaUrl?: string;
    embeddingModel?: string;
  }) {
    this.qdrantClient = new QdrantClient({
      url: options?.qdrantUrl || process.env.QDRANT_URL || 'http://localhost:6333'
    });
    this.ollamaBaseUrl = options?.ollamaUrl || process.env.OLLAMA_URL || 'http://localhost:11434';
    this.embeddingModel = options?.embeddingModel || 'nomic-embed-text';
  }

  /**
   * Generate embedding using Ollama
   */
  async generateEmbedding(text: string): Promise<number[]> {
    try {
      const response = await fetch(`${this.ollamaBaseUrl}/api/embeddings`, {
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
      return data.embedding;
    } catch (error) {
      console.error('Embedding generation failed:', error);
      throw new Error(`Failed to generate embedding: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Perform vector similarity search
   */
  async search(query: string, options: VectorSearchOptions = {}): Promise<SearchResponse> {
    const startTime = Date.now();
    
    try {
      // Generate query embedding
      const queryEmbedding = await this.generateEmbedding(query);

      // Set defaults
      const {
        maxResults = 10,
        threshold = 0.7,
        collection = 'legal_documents',
        includeMetadata = true,
        filter
      } = options;

      // Search in Qdrant
      const searchResults = await this.qdrantClient.search(collection, {
        vector: queryEmbedding,
        limit: maxResults,
        score_threshold: threshold,
        with_payload: includeMetadata,
        filter: filter ? this.buildQdrantFilter(filter) : undefined
      });

      // Transform results
      const results: VectorSearchResult[] = searchResults.map(result => ({
        id: result.id.toString(),
        content: result.payload?.content || result.payload?.text || '',
        score: result.score,
        metadata: includeMetadata ? result.payload : undefined
      }));

      return {
        success: true,
        results,
        totalResults: results.length,
        queryTime: Date.now() - startTime,
        model: this.embeddingModel
      };

    } catch (error) {
      console.error('Vector search failed:', error);
      return {
        success: false,
        results: [],
        totalResults: 0,
        queryTime: Date.now() - startTime,
        model: this.embeddingModel
      };
    }
  }

  /**
   * Store document with embedding
   */
  async storeDocument(
    id: string,
    content: string,
    metadata: Record<string, any> = {},
    collection: string = 'legal_documents'
  ): Promise<boolean> {
    try {
      // Generate embedding
      const embedding = await this.generateEmbedding(content);

      // Store in Qdrant
      await this.qdrantClient.upsert(collection, {
        wait: true,
        points: [{
          id,
          vector: embedding,
          payload: {
            content,
            ...metadata,
            stored_at: new Date().toISOString()
          }
        }]
      });

      return true;
    } catch (error) {
      console.error('Document storage failed:', error);
      return false;
    }
  }

  /**
   * Create collection if it doesn't exist
   */
  async ensureCollection(
    collectionName: string,
    vectorSize: number = 384 // Default for nomic-embed-text
  ): Promise<boolean> {
    try {
      // Check if collection exists
      const collections = await this.qdrantClient.getCollections();
      const exists = collections.collections.some(c => c.name === collectionName);

      if (!exists) {
        await this.qdrantClient.createCollection(collectionName, {
          vectors: {
            size: vectorSize,
            distance: 'Cosine'
          },
          optimizers_config: {
            default_segment_number: 2
          }
        });
        console.log(`✅ Created Qdrant collection: ${collectionName}`);
      }

      return true;
    } catch (error) {
      console.error(`Failed to ensure collection ${collectionName}:`, error);
      return false;
    }
  }

  /**
   * Health check for all services
   */
  async healthCheck(): Promise<{
    ollama: boolean;
    qdrant: boolean;
    overall: boolean;
  }> {
    const health = {
      ollama: false,
      qdrant: false,
      overall: false
    };

    // Check Ollama
    try {
      const response = await fetch(`${this.ollamaBaseUrl}/api/tags`);
      health.ollama = response.ok;
    } catch (error) {
      console.warn('Ollama health check failed:', error);
    }

    // Check Qdrant
    try {
      await this.qdrantClient.getCollections();
      health.qdrant = true;
    } catch (error) {
      console.warn('Qdrant health check failed:', error);
    }

    health.overall = health.ollama && health.qdrant;
    return health;
  }

  /**
   * Get available Ollama models
   */
  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.ollamaBaseUrl}/api/tags`);
      if (!response.ok) return [];

      const data = await response.json();
      return data.models?.map((m: any) => m.name) || [];
    } catch (error) {
      console.error('Failed to get Ollama models:', error);
      return [];
    }
  }

  /**
   * Build Qdrant filter from simple key-value pairs
   */
  private buildQdrantFilter(filter: Record<string, any>) {
    const must = Object.entries(filter).map(([key, value]) => ({
      key,
      match: { value }
    }));

    return { must };
  }
}

// Export singleton instance
export const vectorSearchService = new RealVectorSearchService();