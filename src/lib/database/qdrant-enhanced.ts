// Enhanced Qdrant Vector Database Service
// Provides comprehensive vector storage and search capabilities

export interface QdrantPoint {
  id: string | number;
  vector: number[];
  payload: Record<string, any>;
}

export interface QdrantSearchResult {
  id: string | number;
  score: number;
  payload: Record<string, any>;
  vector?: number[];
}

export interface DocumentUpsertRequest {
  id: string;
  vector: number[];
  payload: {
    documentId: string;
    title: string;
    documentType: string;
    jurisdiction: string;
    practiceArea: string;
    content: string;
    metadata: Record<string, any>;
    timestamp: number;
  };
}

class EnhancedQdrantManager {
  private baseUrl: string;
  private collection: string;
  private connected = false;
  private client: unknown;

  constructor(options: {
    baseUrl?: string;
    collection?: string;
    apiKey?: string;
  } = {}) {
    this.baseUrl = options.baseUrl || process.env.QDRANT_URL || 'http://localhost:6333';
    this.collection = options.collection || 'legal_documents';
    
    // Initialize client if Qdrant client library is available
    this.initializeClient(options.apiKey);
  }

  private async initializeClient(apiKey?: string): Promise<void> {
    try {
      // Try to use @qdrant/js-client-rest if available
      const { QdrantClient } = await import('@qdrant/js-client-rest');
      this.client = new QdrantClient({
        url: this.baseUrl,
        apiKey: apiKey || process.env.QDRANT_API_KEY
      });
      console.log('Qdrant client initialized');
    } catch (error) {
      console.warn('Qdrant client library not available, using fetch API');
      this.client = null;
    }
  }

  /**
   * Connect to Qdrant and ensure collection exists
   */
  async connect(): Promise<boolean> {
    try {
      // Test connection
      const response = await this.fetchWithRetry(`${this.baseUrl}/collections`);
      if (!response.ok) {
        throw new Error(`Qdrant connection failed: ${response.status}`);
      }

      this.connected = true;
      console.log('Connected to Qdrant');
      
      // Ensure collection exists
      await this.ensureCollection();
      return true;

    } catch (error) {
      console.error('Failed to connect to Qdrant:', error);
      this.connected = false;
      return false;
    }
  }

  /**
   * Ensure collection exists with proper configuration
   */
  async ensureCollection(): Promise<void> {
    try {
      const response = await this.fetchWithRetry(`${this.baseUrl}/collections/${this.collection}`);
      
      if (!response.ok) {
        if (response.status === 404) {
          console.log(`Creating collection: ${this.collection}`);
          await this.createCollection();
        } else {
          throw new Error(`Collection check failed: ${response.status}`);
        }
      } else {
        console.log(`Collection ${this.collection} exists`);
      }
    } catch (error) {
      console.error('Failed to ensure collection:', error);
      throw error;
    }
  }

  /**
   * Create a new collection with optimized settings
   */
  async createCollection(): Promise<boolean> {
    try {
      const collectionConfig = {
        vectors: {
          size: 384, // nomic-embed-text dimensions
          distance: 'Cosine'
        },
        optimizers_config: {
          default_segment_number: 2
        },
        replication_factor: 1,
        shard_number: 1
      };

      const response = await this.fetchWithRetry(
        `${this.baseUrl}/collections/${this.collection}`,
        {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(collectionConfig)
        }
      );

      if (response.ok) {
        console.log(`Successfully created collection: ${this.collection}`);
        return true;
      } else {
        const errorText = await response.text();
        throw new Error(`Failed to create collection: ${response.status} - ${errorText}`);
      }

    } catch (error) {
      console.error('Collection creation failed:', error);
      return false;
    }
  }

  /**
   * Upsert document with vector and metadata
   */
  async upsertDocument(request: DocumentUpsertRequest): Promise<boolean> {
    try {
      if (!this.connected) {
        await this.connect();
      }

      const point: QdrantPoint = {
        id: request.id,
        vector: request.vector,
        payload: request.payload
      };

      return await this.upsertPoints([point]);

    } catch (error) {
      console.error('Document upsert failed:', error);
      return false;
    }
  }

  /**
   * Upsert multiple points
   */
  async upsertPoints(points: QdrantPoint[]): Promise<boolean> {
    try {
      if (!this.connected) {
        await this.connect();
      }

      if (this.client) {
        // Use official client if available
        const result = await this.client.upsert(this.collection, {
          wait: true,
          points: points
        });
        return result.status === 'ok';
      } else {
        // Fallback to fetch API
        const response = await this.fetchWithRetry(
          `${this.baseUrl}/collections/${this.collection}/points`,
          {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              points: points,
              wait: true
            })
          }
        );

        if (response.ok) {
          console.log(`Upserted ${points.length} points`);
          return true;
        } else {
          const errorText = await response.text();
          console.error(`Upsert failed: ${response.status} - ${errorText}`);
          return false;
        }
      }

    } catch (error) {
      console.error('Points upsert failed:', error);
      return false;
    }
  }

  /**
   * Search for similar vectors
   */
  async search(
    vector: number[], 
    options: {
      limit?: number;
      scoreThreshold?: number;
      filter?: Record<string, any>;
      withVector?: boolean;
      withPayload?: boolean;
    } = {}
  ): Promise<QdrantSearchResult[]> {
    try {
      if (!this.connected) {
        await this.connect();
      }

      const searchRequest = {
        vector: vector,
        limit: options.limit || 10,
        score_threshold: options.scoreThreshold || 0.0,
        with_payload: options.withPayload !== false,
        with_vector: options.withVector || false,
        ...(options.filter && { filter: options.filter })
      };

      if (this.client) {
        // Use official client
        const results = await this.client.search(this.collection, searchRequest);
        return results.map(result => ({
          id: result.id,
          score: result.score,
          payload: result.payload || {},
          vector: result.vector
        }));
      } else {
        // Fallback to fetch API
        const response = await this.fetchWithRetry(
          `${this.baseUrl}/collections/${this.collection}/points/search`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(searchRequest)
          }
        );

        if (response.ok) {
          const data = await response.json();
          return data.result || [];
        } else {
          console.error(`Search failed: ${response.status}`);
          return [];
        }
      }

    } catch (error) {
      console.error('Vector search failed:', error);
      return [];
    }
  }

  /**
   * Search similar documents by text query
   */
  async searchDocuments(
    queryVector: number[],
    options: {
      limit?: number;
      documentType?: string;
      jurisdiction?: string;
      practiceArea?: string;
      scoreThreshold?: number;
    } = {}
  ): Promise<Array<{
    documentId: string;
    title: string;
    score: number;
    documentType: string;
    jurisdiction: string;
    practiceArea: string;
    content: string;
    metadata: Record<string, any>;
  }>> {
    try {
      // Build filter based on options
      const filter: Record<string, any> = {};
      
      if (options.documentType) {
        filter.documentType = { $eq: options.documentType };
      }
      
      if (options.jurisdiction) {
        filter.jurisdiction = { $eq: options.jurisdiction };
      }
      
      if (options.practiceArea) {
        filter.practiceArea = { $eq: options.practiceArea };
      }

      const searchOptions = {
        limit: options.limit || 10,
        scoreThreshold: options.scoreThreshold || 0.3,
        filter: Object.keys(filter).length > 0 ? filter : undefined,
        withPayload: true
      };

      const results = await this.search(queryVector, searchOptions);
      
      return results.map(result => ({
        documentId: result.payload.documentId,
        title: result.payload.title,
        score: result.score,
        documentType: result.payload.documentType,
        jurisdiction: result.payload.jurisdiction,
        practiceArea: result.payload.practiceArea,
        content: result.payload.content,
        metadata: result.payload.metadata || {}
      }));

    } catch (error) {
      console.error('Document search failed:', error);
      return [];
    }
  }

  /**
   * Get point by ID
   */
  async getPoint(id: string): Promise<QdrantPoint | null> {
    try {
      if (!this.connected) {
        await this.connect();
      }

      if (this.client) {
        const result = await this.client.retrieve(this.collection, {
          ids: [id],
          with_payload: true,
          with_vector: true
        });
        
        if (result.length > 0) {
          const point = result[0];
          return {
            id: point.id,
            vector: point.vector || [],
            payload: point.payload || {}
          };
        }
        return null;
      } else {
        const response = await this.fetchWithRetry(
          `${this.baseUrl}/collections/${this.collection}/points/${id}`,
          {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' }
          }
        );

        if (response.ok) {
          const data = await response.json();
          return data.result;
        }
        return null;
      }

    } catch (error) {
      console.error(`Failed to get point ${id}:`, error);
      return null;
    }
  }

  /**
   * Delete point by ID
   */
  async deletePoint(id: string): Promise<boolean> {
    try {
      if (!this.connected) {
        await this.connect();
      }

      if (this.client) {
        const result = await this.client.delete(this.collection, {
          wait: true,
          points: [id]
        });
        return result.status === 'ok';
      } else {
        const response = await this.fetchWithRetry(
          `${this.baseUrl}/collections/${this.collection}/points/delete`,
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              points: [id],
              wait: true
            })
          }
        );

        return response.ok;
      }

    } catch (error) {
      console.error(`Failed to delete point ${id}:`, error);
      return false;
    }
  }

  /**
   * Delete multiple documents
   */
  async deleteDocuments(documentIds: string[]): Promise<number> {
    try {
      let deletedCount = 0;
      
      for (const id of documentIds) {
        if (await this.deletePoint(id)) {
          deletedCount++;
        }
      }

      console.log(`Deleted ${deletedCount}/${documentIds.length} documents`);
      return deletedCount;

    } catch (error) {
      console.error('Batch delete failed:', error);
      return 0;
    }
  }

  /**
   * Get collection information
   */
  async getCollectionInfo(): Promise<any> {
    try {
      if (!this.connected) {
        await this.connect();
      }

      const response = await this.fetchWithRetry(
        `${this.baseUrl}/collections/${this.collection}`
      );

      if (response.ok) {
        const data = await response.json();
        return data.result;
      }
      return null;

    } catch (error) {
      console.error('Failed to get collection info:', error);
      return null;
    }
  }

  /**
   * Clear entire collection
   */
  async clearCollection(): Promise<boolean> {
    try {
      const response = await this.fetchWithRetry(
        `${this.baseUrl}/collections/${this.collection}`,
        { method: 'DELETE' }
      );

      if (response.ok) {
        console.log('Collection cleared');
        await this.createCollection();
        return true;
      }
      return false;

    } catch (error) {
      console.error('Failed to clear collection:', error);
      return false;
    }
  }

  /**
   * Get service health status
   */
  async getHealthStatus(): Promise<{
    connected: boolean;
    collection: string;
    vectorCount?: number;
    collectionStatus?: string;
    error?: string;
  }> {
    try {
      if (!this.connected) {
        await this.connect();
      }

      const info = await this.getCollectionInfo();
      
      return {
        connected: this.connected,
        collection: this.collection,
        vectorCount: info?.vectors_count || 0,
        collectionStatus: info?.status || 'unknown'
      };

    } catch (error: unknown) {
      return {
        connected: false,
        collection: this.collection,
        error: error.message
      };
    }
  }

  /**
   * Batch operations for bulk document processing
   */
  async batchUpsertDocuments(documents: DocumentUpsertRequest[]): Promise<{
    success: number;
    failed: number;
    errors: string[];
  }> {
    const results = { success: 0, failed: 0, errors: [] as string[] };
    const batchSize = 100; // Process in batches to avoid memory issues

    try {
      for (let i = 0; i < documents.length; i += batchSize) {
        const batch = documents.slice(i, i + batchSize);
        const points: QdrantPoint[] = batch.map(doc => ({
          id: doc.id,
          vector: doc.vector,
          payload: doc.payload
        }));

        try {
          const success = await this.upsertPoints(points);
          if (success) {
            results.success += batch.length;
          } else {
            results.failed += batch.length;
            results.errors.push(`Batch ${i / batchSize + 1} failed`);
          }
        } catch (error: unknown) {
          results.failed += batch.length;
          results.errors.push(`Batch ${i / batchSize + 1}: ${error.message}`);
        }
      }

      console.log(`Batch upsert completed: ${results.success} success, ${results.failed} failed`);
      return results;

    } catch (error: unknown) {
      results.errors.push(`Batch operation failed: ${error.message}`);
      return results;
    }
  }

  /**
   * Helper method for fetch with retry logic
   */
  private async fetchWithRetry(
    url: string, 
    options: RequestInit = {}, 
    maxRetries = 3
  ): Promise<Response> {
    let lastError: Error;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const response = await fetch(url, {
          ...options,
          headers: {
            'Content-Type': 'application/json',
            ...options.headers
          }
        });

        if (response.ok || attempt === maxRetries) {
          return response;
        }

        // Wait before retrying (exponential backoff)
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));

      } catch (error: unknown) {
        lastError = error;
        if (attempt === maxRetries) {
          throw error;
        }
        
        // Wait before retrying
        await new Promise(resolve => setTimeout(resolve, Math.pow(2, attempt) * 1000));
      }
    }

    throw lastError!;
  }

  /**
   * Get connection status
   */
  getStatus(): {
    connected: boolean;
    baseUrl: string;
    collection: string;
    clientType: string;
  } {
    return {
      connected: this.connected,
      baseUrl: this.baseUrl,
      collection: this.collection,
      clientType: this.client ? 'official' : 'fetch'
    };
  }
}

// Export singleton instance
export const qdrantManager = new EnhancedQdrantManager();

// Export class for testing/instantiation
export { EnhancedQdrantManager };

// Legacy compatibility exports
export default qdrantManager;
