import { OLLAMA_API_URL } from "$env/static/private";

/**
 * Embedding service for RAG (Retrieval-Augmented Generation) functionality
 * Uses Ollama's nomic-embed-text model for generating embeddings
 */

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

interface HealthStatus {
  status: 'healthy' | 'unhealthy';
  ollamaUrl: string;
  embeddingModel: string;
  modelAvailable?: boolean;
  dimensions?: number;
  timestamp: string;
  error?: string;
}

class EmbeddingService {
  private ollamaUrl: string;
  private embeddingModel: string;
  private dimensions: number;

  constructor() {
    this.ollamaUrl = OLLAMA_API_URL || "http://localhost:11434";
    this.embeddingModel = "nomic-embed-text";
    this.dimensions = 384; // nomic-embed-text embedding dimensions
  }

  /**
   * Generate embeddings for a single text
   * @param text - Text to generate embeddings for
   * @returns Array of embedding values
   */
  async generateEmbedding(text: string): Promise<number[]> {
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
   * @param texts - Array of texts to generate embeddings for
   * @returns Array of embedding arrays
   */
  async generateBatchEmbeddings(texts: string[]): Promise<number[][]> {
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
   * @param embedding1 - First embedding vector
   * @param embedding2 - Second embedding vector
   * @returns Cosine similarity score (0-1)
   */
  cosineSimilarity(embedding1: number[], embedding2: number[]): number {
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
   * @param text - Text to chunk
   * @param options - Chunking options
   * @returns Array of text chunks with metadata
   */
  chunkText(text: string, options: ChunkOptions = {}): TextChunk[] {
    const { chunkSize = 500, chunkOverlap = 100, separator = "\n\n" } = options;

    if (!text || text.trim().length === 0) {
      return [];
    }

    const chunks: TextChunk[] = [];
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
   * @param text - Text to estimate tokens for
   * @returns Estimated token count
   */
  estimateTokenCount(text: string): number {
    // Rough approximation: 1 token ≈ 4 characters for English text
    return Math.ceil(text.length / 4);
  }

  /**
   * Preprocess text for better embeddings
   * @param text - Text to preprocess
   * @returns Preprocessed text
   */
  preprocessText(text: string): string {
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
   * @returns Service health status
   */
  async healthCheck(): Promise<HealthStatus> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/tags`);

      if (!response.ok) {
        throw new Error(`Ollama API not responding: ${response.status}`);
      }

      const data = await response.json();
      const hasEmbeddingModel = data.models?.some((model: unknown) =>
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
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      return {
        status: "unhealthy",
        error: errorMessage,
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
