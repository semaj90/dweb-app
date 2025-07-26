import { OLLAMA_URL } from "$env/static/private";

/**
 * Embedding service for RAG (Retrieval-Augmented Generation) functionality
 * Uses Ollama's nomic-embed-text model for generating embeddings
 */

class EmbeddingService {
  constructor() {
    this.ollamaUrl = OLLAMA_URL || "http://localhost:11434";
    this.embeddingModel = "nomic-embed-text";
    this.dimensions = 384; // nomic-embed-text embedding dimensions
  }

  /**
   * Generate embeddings for a single text
   * @param {string} text - Text to generate embeddings for
   * @returns {Promise<number[]>} - Array of embedding values
   */
  async generateEmbedding(text) {
    try {
      if (!text || text.trim().length === 0) {
        throw new Error("Text cannot be empty");
      }

      const response = await fetch(`${this.ollamaUrl}/api/embeddings`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: this.embeddingModel,
          prompt: text.trim(),
        }),
      });

      if (!response.ok) {
        throw new Error(
          `Ollama embedding API error: ${response.status} ${response.statusText}`
        );
      }

      const data = await response.json();

      if (!data.embedding || !Array.isArray(data.embedding)) {
        throw new Error("Invalid embedding response from Ollama");
      }

      return data.embedding;
    } catch (error) {
      console.error("Error generating embedding:", error);
      throw error;
    }
  }

  /**
   * Generate embeddings for multiple texts in batch
   * @param {string[]} texts - Array of texts to generate embeddings for
   * @returns {Promise<number[][]>} - Array of embedding arrays
   */
  async generateBatchEmbeddings(texts) {
    try {
      const embeddings = await Promise.all(
        texts.map((text) => this.generateEmbedding(text))
      );
      return embeddings;
    } catch (error) {
      console.error("Error generating batch embeddings:", error);
      throw error;
    }
  }

  /**
   * Calculate cosine similarity between two embeddings
   * @param {number[]} embedding1 - First embedding vector
   * @param {number[]} embedding2 - Second embedding vector
   * @returns {number} - Cosine similarity score (0-1)
   */
  cosineSimilarity(embedding1, embedding2) {
    if (embedding1.length !== embedding2.length) {
      throw new Error("Embeddings must have the same dimensions");
    }

    let dotProduct = 0;
    let magnitude1 = 0;
    let magnitude2 = 0;

    for (let i = 0; i < embedding1.length; i++) {
      dotProduct += embedding1[i] * embedding2[i];
      magnitude1 += embedding1[i] * embedding1[i];
      magnitude2 += embedding2[i] * embedding2[i];
    }

    magnitude1 = Math.sqrt(magnitude1);
    magnitude2 = Math.sqrt(magnitude2);

    if (magnitude1 === 0 || magnitude2 === 0) {
      return 0;
    }

    return dotProduct / (magnitude1 * magnitude2);
  }

  /**
   * Chunk text into smaller pieces for embedding
   * @param {string} text - Text to chunk
   * @param {Object} options - Chunking options
   * @returns {Array<Object>} - Array of text chunks with metadata
   */
  chunkText(text, options = {}) {
    const { chunkSize = 500, chunkOverlap = 100, separator = "\n\n" } = options;

    if (!text || text.trim().length === 0) {
      return [];
    }

    const chunks = [];
    let startIndex = 0;

    // First try to split by separator (paragraphs)
    const paragraphs = text.split(separator).filter((p) => p.trim().length > 0);

    let currentChunk = "";
    let currentIndex = 0;

    for (const paragraph of paragraphs) {
      // If adding this paragraph would exceed chunk size
      if (
        currentChunk.length + paragraph.length > chunkSize &&
        currentChunk.length > 0
      ) {
        // Save current chunk
        chunks.push({
          content: currentChunk.trim(),
          startOffset: currentIndex - currentChunk.length,
          endOffset: currentIndex,
          tokens: this.estimateTokenCount(currentChunk),
          chunkIndex: chunks.length,
        });

        // Start new chunk with overlap
        const overlapText = currentChunk.slice(-chunkOverlap);
        currentChunk = overlapText + paragraph;
        currentIndex += paragraph.length + separator.length;
      } else {
        // Add paragraph to current chunk
        if (currentChunk.length > 0) {
          currentChunk += separator;
        }
        currentChunk += paragraph;
        currentIndex += paragraph.length + separator.length;
      }
    }

    // Add final chunk if it exists
    if (currentChunk.trim().length > 0) {
      chunks.push({
        content: currentChunk.trim(),
        startOffset: currentIndex - currentChunk.length,
        endOffset: currentIndex,
        tokens: this.estimateTokenCount(currentChunk),
        chunkIndex: chunks.length,
      });
    }

    return chunks;
  }

  /**
   * Estimate token count for text (rough approximation)
   * @param {string} text - Text to estimate tokens for
   * @returns {number} - Estimated token count
   */
  estimateTokenCount(text) {
    // Rough approximation: 1 token ≈ 4 characters for English text
    return Math.ceil(text.length / 4);
  }

  /**
   * Preprocess text for better embeddings
   * @param {string} text - Text to preprocess
   * @returns {string} - Preprocessed text
   */
  preprocessText(text) {
    if (!text) return "";

    return (
      text
        // Remove excessive whitespace
        .replace(/\s+/g, " ")
        // Remove special characters that might interfere
        .replace(/[^\w\s\.\,\!\?\;\:\-\(\)]/g, "")
        // Trim
        .trim()
    );
  }

  /**
   * Health check for the embedding service
   * @returns {Promise<Object>} - Service health status
   */
  async healthCheck() {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);

      if (!response.ok) {
        throw new Error(`Ollama API not responding: ${response.status}`);
      }

      const data = await response.json();
      const hasEmbeddingModel = data.models?.some((model) =>
        model.name.includes(this.embeddingModel)
      );

      return {
        status: "healthy",
        ollamaUrl: this.ollamaUrl,
        embeddingModel: this.embeddingModel,
        modelAvailable: hasEmbeddingModel,
        dimensions: this.dimensions,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      return {
        status: "unhealthy",
        error: error.message,
        ollamaUrl: this.ollamaUrl,
        embeddingModel: this.embeddingModel,
        timestamp: new Date().toISOString(),
      };
    }
  }
}

// Export singleton instance
export const embeddingService = new EmbeddingService();
export default embeddingService;
