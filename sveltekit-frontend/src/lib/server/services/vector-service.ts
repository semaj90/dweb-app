/**
 * Production Vector Service - Real Implementation
 * Integrates Redis Vector DB, Qdrant, and Ollama for production use
 */

import { redisVectorService } from "../../../services/redis-vector-service";

export interface EmbeddingOptions {
  contentType?: string;
  metadata?: Record<string, any>;
  model?: string;
  userId?: string;
  caseId?: string;
  conversationId?: string;
}

export class VectorService {
  private static ollamaUrl = "http://localhost:11434";
  private static embeddingModel = "nomic-embed-text";

  /**
   * Generate embedding using Ollama
   */
  static async generateEmbedding(
    content: string,
    options: EmbeddingOptions = {}
  ): Promise<number[]> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/embeddings`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: options.model || this.embeddingModel,
          prompt: content,
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status}`);
      }

      const data = await response.json();
      return data.embedding;
    } catch (error) {
      console.error("Error generating embedding:", error);
      throw error;
    }
  }

  /**
   * Generate embedding with metadata
   */
  static async generateEmbeddingWithMetadata(
    content: string,
    options: EmbeddingOptions = {}
  ): Promise<{ embedding: number[]; model: string }> {
    const embedding = await this.generateEmbedding(content, options);
    return {
      embedding,
      model: options.model || this.embeddingModel,
    };
  }

  /**
   * Store evidence vector in Redis
   */
  static async storeEvidenceVector(evidence: {
    id: string;
    content: string;
    metadata?: any;
  }): Promise<void> {
    const embedding = await this.generateEmbedding(evidence.content);
    
    await redisVectorService.storeDocument({
      id: `evidence:${evidence.id}`,
      embedding,
      content: evidence.content,
      metadata: {
        type: "evidence",
        ...evidence.metadata,
      },
    });
  }

  /**
   * Store case embedding
   */
  static async storeCaseEmbedding(data: {
    caseId: string;
    content: string;
    metadata?: any;
  }): Promise<void> {
    const embedding = await this.generateEmbedding(data.content);
    
    await redisVectorService.storeDocument({
      id: `case:${data.caseId}`,
      embedding,
      content: data.content,
      metadata: {
        type: "case",
        caseId: data.caseId,
        ...data.metadata,
      },
    });
  }

  /**
   * Store chat embedding
   */
  static async storeChatEmbedding(data: {
    conversationId: string;
    messageId: string;
    content: string;
    userId?: string;
  }): Promise<void> {
    const embedding = await this.generateEmbedding(data.content);
    
    await redisVectorService.storeDocument({
      id: `chat:${data.conversationId}:${data.messageId}`,
      embedding,
      content: data.content,
      metadata: {
        type: "chat",
        conversationId: data.conversationId,
        userId: data.userId,
      },
    });
  }

  /**
   * Find similar vectors using Redis search
   */
  static async findSimilar(
    embedding: number[],
    options: {
      limit?: number;
      threshold?: number;
      type?: string;
    } = {}
  ): Promise<any[]> {
    const results = await redisVectorService.searchSimilar(embedding, {
      topK: options.limit || 10,
      threshold: options.threshold || 0.7,
      filter: options.type ? { type: options.type } : undefined,
    });

    return results.map(result => ({
      id: result.id,
      score: result.score,
      content: result.content,
      metadata: result.metadata,
    }));
  }

  /**
   * Semantic search with text query
   */
  static async semanticSearch(
    query: string,
    options: {
      limit?: number;
      threshold?: number;
      type?: string;
    } = {}
  ): Promise<any[]> {
    const queryEmbedding = await this.generateEmbedding(query);
    return this.findSimilar(queryEmbedding, options);
  }

  /**
   * Store document with automatic embedding
   */
  static async storeDocument(
    documentId: string,
    documentType: string,
    text: string,
    metadata: any = {}
  ): Promise<any> {
    const embedding = await this.generateEmbedding(text);
    
    await redisVectorService.storeDocument({
      id: `doc:${documentId}`,
      embedding,
      content: text,
      metadata: {
        type: documentType,
        documentId,
        ...metadata,
      },
    });

    return { id: documentId, type: documentType };
  }

  /**
   * Analyze document using Ollama
   */
  static async analyzeDocument(
    text: string,
    analysisType: string
  ): Promise<any> {
    try {
      const response = await fetch(`${this.ollamaUrl}/api/generate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: "gemma3-legal",
          prompt: `Analyze this document for ${analysisType}:\n\n${text}\n\nProvide a structured analysis:`,
          stream: false,
        }),
      });

      const data = await response.json();
      return {
        analysis: data.response,
        type: analysisType,
        timestamp: new Date().toISOString(),
      };
    } catch (error) {
      console.error("Error analyzing document:", error);
      return {
        analysis: "Analysis failed",
        error: error.message,
      };
    }
  }

  /**
   * Search documents with semantic similarity
   */
  static async search(
    query: string,
    options: {
      limit?: number;
      threshold?: number;
      type?: string;
    } = {}
  ): Promise<any[]> {
    return this.semanticSearch(query, options);
  }

  /**
   * Find similar documents to a given document
   */
  static async findSimilarDocuments(
    documentId: string,
    limit: number = 10
  ): Promise<any[]> {
    try {
      const doc = await redisVectorService.getDocument(`doc:${documentId}`);
      if (!doc) {
        return [];
      }

      return this.findSimilar(doc.embedding, {
        limit,
        threshold: 0.7,
      });
    } catch (error) {
      console.error("Error finding similar documents:", error);
      return [];
    }
  }

  /**
   * Store user embedding (legacy compatibility)
   */
  static async storeUserEmbedding(
    userId: string,
    content: string,
    embedding: number[],
    options: EmbeddingOptions = {}
  ): Promise<string> {
    await redisVectorService.storeDocument({
      id: `user:${userId}:${Date.now()}`,
      embedding,
      content,
      metadata: {
        type: "user_content",
        userId,
        ...options.metadata,
      },
    });

    return userId;
  }

  /**
   * Get user embeddings (legacy compatibility)
   */
  static async getUserEmbeddings(userId: string): Promise<any[]> {
    const results = await redisVectorService.searchSimilar(
      new Array(384).fill(0), // Dummy embedding for filtering
      {
        topK: 100,
        threshold: 0,
        filter: { userId },
      }
    );

    return results.map(result => ({
      userId,
      content: result.content,
      embedding: JSON.stringify(result.metadata.embedding || []),
      metadata: result.metadata,
      createdAt: result.metadata.timestamp,
    }));
  }

  /**
   * Update evidence metadata
   */
  static async updateEvidenceMetadata(
    evidenceId: string,
    metadata: any
  ): Promise<void> {
    const doc = await redisVectorService.getDocument(`evidence:${evidenceId}`);
    if (doc) {
      doc.metadata = { ...doc.metadata, ...metadata };
      await redisVectorService.storeDocument(doc);
    }
  }

  /**
   * Delete evidence vector
   */
  static async deleteEvidenceVector(evidenceId: string): Promise<void> {
    await redisVectorService.deleteDocument(`evidence:${evidenceId}`);
  }

  /**
   * Simple similarity search (legacy compatibility)
   */
  static async searchSimilar(
    query: string,
    options: {
      limit?: number;
      threshold?: number;
    } = {}
  ): Promise<any[]> {
    return this.semanticSearch(query, options);
  }

  /**
   * Health check
   */
  static async healthCheck(): Promise<boolean> {
    return redisVectorService.healthCheck();
  }
}

export default VectorService;