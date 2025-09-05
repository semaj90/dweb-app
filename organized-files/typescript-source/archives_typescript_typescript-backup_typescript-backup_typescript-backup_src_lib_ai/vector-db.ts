// Vector Database Service
// Manages vector storage and similarity search

export class VectorDB {
  private vectors: Map<string, { vector: number[]; metadata: any }>;
  private dimension: number;

  constructor(dimension: number = 384) {
    this.vectors = new Map();
    this.dimension = dimension;
  }

  /**
   * Add vector to database
   */
  async add(id: string, vector: number[], metadata: any = {}): Promise<any> {
    if (vector.length !== this.dimension) {
      throw new Error(`Vector dimension mismatch. Expected ${this.dimension}, got ${vector.length}`);
    }

    this.vectors.set(id, { vector, metadata });
  }

  /**
   * Search by vector with options
   */
  async searchByVector(queryVector: number[], options?: { 
    limit?: number; 
    threshold?: number; 
    filter?: Record<string, any>;
  }): Promise<Array<{
    id: string;
    score: number;
    document: { id: string; content: string; metadata: any };
  }>> {
    const { limit = 5, threshold = 0.0, filter = {} } = options || {};
    
    const searchResults = await this.search(queryVector, limit);
    
    // Filter by threshold and transform to expected format
    return searchResults
      .filter(result => result.score >= threshold)
      .map(result => ({
        id: result.id,
        score: result.score,
        document: {
          id: result.id,
          content: (result.metadata as any)?.content || '',
          metadata: result.metadata
        }
      }));
  }

  /**
   * Store document with vector
   */
  async storeDocument(id: string, content: string, vector: number[], metadata: any = {}): Promise<any> {
    await this.add(id, vector, { content, ...(typeof metadata === 'object' && metadata !== null ? metadata : {}) });
  }

  /**
   * Search for similar vectors
   */
  async search(queryVector: number[], topK: number = 5): Promise<Array<{
    id: string;
    score: number;
    metadata: any;
  }>> {
    if (queryVector.length !== this.dimension) {
      throw new Error(`Query vector dimension mismatch. Expected ${this.dimension}, got ${queryVector.length}`);
    }

    const results: Array<{ id: string; score: number; metadata: any }> = [];

    // Calculate cosine similarity with all vectors
    for (const [id, entry] of Array.from(this.vectors.entries())) {
      const score = this.cosineSimilarity(queryVector, entry.vector);
      results.push({ id, score, metadata: entry.metadata });
    }

    // Sort by score and return top K
    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    return denominator === 0 ? 0 : dotProduct / denominator;
  }

  /**
   * Get vector by ID
   */
  async get(id: string): Promise<{ vector: number[]; metadata: any } | null> {
    return this.vectors.get(id) || null;
  }

  /**
   * Delete vector by ID
   */
  async delete(id: string): Promise<boolean> {
    return this.vectors.delete(id);
  }

  /**
   * Clear all vectors
   */
  async clear(): Promise<any> {
    this.vectors.clear();
  }

  /**
   * Get database statistics
   */
  getStats() {
    return {
      count: this.vectors.size,
      dimension: this.dimension
    };
  }
}

// Export singleton instance
export const vectorDB = new VectorDB();

// Export for module compatibility
export default vectorDB;
