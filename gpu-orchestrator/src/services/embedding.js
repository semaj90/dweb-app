/**
 * Embedding Service - GPU-accelerated text embeddings
 */
export class EmbeddingService {
  constructor(ollamaService, cudaWorkerPool) {
    this.ollama = ollamaService;
    this.cudaPool = cudaWorkerPool;
    this.embeddingCache = new Map();
    this.defaultModel = 'nomic-embed-text';
  }

  async generateEmbedding(text, options = {}) {
    const { model = this.defaultModel, useCuda = false, useCache = true } = options;
    
    // Check cache first
    const cacheKey = `${model}:${this.hashText(text)}`;
    if (useCache && this.embeddingCache.has(cacheKey)) {
      return this.embeddingCache.get(cacheKey);
    }

    try {
      let embedding;
      
      if (useCuda && this.cudaPool.enabled) {
        // Use GPU acceleration
        embedding = await this.generateWithCuda(text, model);
      } else {
        // Use Ollama directly
        embedding = await this.ollama.generateEmbedding(text, model);
      }

      // Cache the result
      if (useCache) {
        this.embeddingCache.set(cacheKey, embedding);
        
        // Limit cache size
        if (this.embeddingCache.size > 1000) {
          const firstKey = this.embeddingCache.keys().next().value;
          this.embeddingCache.delete(firstKey);
        }
      }

      return embedding;
    } catch (error) {
      console.error('Embedding generation failed:', error);
      throw error;
    }
  }

  async generateWithCuda(text, model) {
    try {
      const job = {
        type: 'embedding',
        text,
        model,
        timestamp: Date.now()
      };

      const result = await this.cudaPool.processJob(job);
      
      // For now, fallback to Ollama if CUDA processing doesn't return embedding
      if (!result.embedding) {
        return await this.ollama.generateEmbedding(text, model);
      }
      
      return result.embedding;
    } catch (error) {
      console.warn('CUDA embedding failed, falling back to Ollama:', error);
      return await this.ollama.generateEmbedding(text, model);
    }
  }

  async generateBatchEmbeddings(texts, options = {}) {
    const { batchSize = 10, ...embeddingOptions } = options;
    const results = [];
    
    for (let i = 0; i < texts.length; i += batchSize) {
      const batch = texts.slice(i, i + batchSize);
      const batchPromises = batch.map(text => 
        this.generateEmbedding(text, embeddingOptions)
      );
      
      const batchResults = await Promise.allSettled(batchPromises);
      results.push(...batchResults);
    }
    
    return results.map((result, index) => ({
      text: texts[index],
      embedding: result.status === 'fulfilled' ? result.value : null,
      error: result.status === 'rejected' ? result.reason : null
    }));
  }

  hashText(text) {
    // Simple hash function for caching
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash.toString(36);
  }

  clearCache() {
    this.embeddingCache.clear();
  }

  getCacheStats() {
    return {
      size: this.embeddingCache.size,
      keys: Array.from(this.embeddingCache.keys()).slice(0, 10) // First 10 keys
    };
  }
}