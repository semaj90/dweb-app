/**
 * Enhanced Embedding Service
 * Production-ready service for generating and managing vector embeddings
 * with Ollama nomic-embed-text model and pgvector storage
 */

import { db } from "./db/index";
import { users, documentEmbeddings, caseEmbeddings } from "./db/schema-unified";
import { eq } from "drizzle-orm";

export interface OllamaEmbeddingResponse {
  embedding: number[];
}

export interface EmbeddingOptions {
  model?: string;
  dimensions?: number;
  normalize?: boolean;
}

export class EmbeddingService {
  private readonly baseUrl: string;
  private readonly model: string;
  private readonly dimensions: number;

  constructor(
    baseUrl = 'http://localhost:11434',
    model = 'nomic-embed-text',
    dimensions = 384
  ) {
    this.baseUrl = baseUrl;
    this.model = model;
    this.dimensions = dimensions;
  }

  /**
   * Generate embedding for text using Ollama
   */
  async generateEmbedding(
    text: string, 
    options: EmbeddingOptions = {}
  ): Promise<number[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: options.model || this.model,
          prompt: text,
        }),
      });

      if (!response.ok) {
        throw new Error(`Ollama API error: ${response.status} ${response.statusText}`);
      }

      const data: OllamaEmbeddingResponse = await response.json();
      
      if (!data.embedding || !Array.isArray(data.embedding)) {
        throw new Error('Invalid embedding response from Ollama');
      }

      let embedding = data.embedding;

      // Normalize vector if requested
      if (options.normalize !== false) {
        embedding = this.normalizeVector(embedding);
      }

      // Ensure dimensions match expected
      if (embedding.length !== (options.dimensions || this.dimensions)) {
        console.warn(`Embedding dimension mismatch: expected ${options.dimensions || this.dimensions}, got ${embedding.length}`);
      }

      return embedding;
    } catch (error: any) {
      console.error('Error generating embedding:', error);
      throw error;
    }
  }

  /**
   * Generate embeddings for multiple texts in batch
   */
  async generateBatchEmbeddings(
    texts: string[],
    options: EmbeddingOptions = {}
  ): Promise<number[][]> {
    const embeddings = await Promise.all(
      texts.map(text => this.generateEmbedding(text, options))
    );
    return embeddings;
  }

  /**
   * Generate and store user profile embedding
   */
  async generateUserProfileEmbedding(userId: string): Promise<void> {
    try {
      // Get user data
      const [user] = await db.select().from(users).where(eq(users.id, userId)).limit(1);
      
      if (!user) {
        throw new Error('User not found');
      }

      // Create profile text for embedding
      const profileParts = [
        user.name,
        user.bio,
        user.firstName,
        user.lastName,
        Array.isArray(user.legalSpecialties) ? user.legalSpecialties.join(' ') : '',
      ].filter(Boolean);

      const profileText = profileParts.join(' ').trim();
      
      if (!profileText) {
        console.warn(`No profile text for user ${userId}, skipping embedding generation`);
        return;
      }

      // Generate embedding
      const embedding = await this.generateEmbedding(profileText);

      // Store in database
      await db.update(users)
        .set({ 
          profileEmbedding: `[${embedding.join(',')}]`, // Store as vector string
          updatedAt: new Date()
        })
        .where(eq(users.id, userId));

      console.log(`Generated profile embedding for user ${userId}`);
    } catch (error: any) {
      console.error(`Error generating user profile embedding for ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Generate and store user preference embedding
   */
  async generateUserPreferenceEmbedding(userId: string): Promise<void> {
    try {
      // Get user data
      const [user] = await db.select().from(users).where(eq(users.id, userId)).limit(1);
      
      if (!user) {
        throw new Error('User not found');
      }

      // Create preference text for embedding
      const preferenceParts = [];
      
      // Add legal specialties
      if (Array.isArray(user.legalSpecialties)) {
        preferenceParts.push(...user.legalSpecialties);
      }

      // Add preferences if it's an object
      if (user.preferences && typeof user.preferences === 'object') {
        const prefs = user.preferences as Record<string, any>;
        Object.entries(prefs).forEach(([key, value]) => {
          if (typeof value === 'string') {
            preferenceParts.push(`${key}: ${value}`);
          } else if (typeof value === 'boolean') {
            preferenceParts.push(`${key}: ${value ? 'enabled' : 'disabled'}`);
          }
        });
      }

      const preferenceText = preferenceParts.join(' ').trim();
      
      if (!preferenceText) {
        console.warn(`No preference text for user ${userId}, skipping embedding generation`);
        return;
      }

      // Generate embedding
      const embedding = await this.generateEmbedding(preferenceText);

      // Store in database
      await db.update(users)
        .set({ 
          preferenceEmbedding: `[${embedding.join(',')}]`, // Store as vector string
          updatedAt: new Date()
        })
        .where(eq(users.id, userId));

      console.log(`Generated preference embedding for user ${userId}`);
    } catch (error: any) {
      console.error(`Error generating user preference embedding for ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Generate document embedding and store in database
   */
  async generateDocumentEmbedding(
    content: string,
    metadata: {
      documentId?: string;
      evidenceId?: string;
      chunkIndex?: number;
      chunkText?: string;
      parentChunkId?: string;
    }
  ): Promise<void> {
    try {
      // Generate embedding
      const embedding = await this.generateEmbedding(content);

      // Store in document_embeddings table
      await db.insert(documentEmbeddings).values({
        documentId: metadata.documentId || null,
        evidenceId: metadata.evidenceId || null,
        content,
        embedding: `[${embedding.join(',')}]`, // Store as vector string
        chunkIndex: metadata.chunkIndex || 0,
        chunkText: metadata.chunkText || content,
        chunkSize: content.length,
        parentChunkId: metadata.parentChunkId || null,
        embeddingModel: this.model,
        metadata: {}
      });

      console.log('Generated and stored document embedding');
    } catch (error: any) {
      console.error('Error generating document embedding:', error);
      throw error;
    }
  }

  /**
   * Generate case embedding and store in database
   */
  async generateCaseEmbedding(caseId: string, content: string): Promise<void> {
    try {
      // Generate embedding
      const embedding = await this.generateEmbedding(content);

      // Store in case_embeddings table
      await db.insert(caseEmbeddings).values({
        caseId,
        content,
        embedding: `[${embedding.join(',')}]`, // Store as vector string
        metadata: {}
      });

      console.log(`Generated and stored case embedding for case ${caseId}`);
    } catch (error: any) {
      console.error(`Error generating case embedding for ${caseId}:`, error);
      throw error;
    }
  }

  /**
   * Calculate cosine similarity between two vectors
   */
  cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
      throw new Error('Vectors must have the same length');
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    
    if (denominator === 0) {
      return 0;
    }

    return dotProduct / denominator;
  }

  /**
   * Normalize vector to unit length
   */
  private normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    
    if (norm === 0) {
      return vector; // Return original if zero vector
    }

    return vector.map(val => val / norm);
  }

  /**
   * Chunk text into smaller pieces for embedding
   */
  chunkText(
    text: string, 
    chunkSize: number = 600, 
    overlap: number = 60
  ): { text: string; index: number }[] {
    const chunks = [];
    let start = 0;
    let index = 0;

    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      const chunk = text.slice(start, end);
      
      chunks.push({
        text: chunk,
        index: index++
      });

      // Move start position considering overlap
      start = end - overlap;
      
      // Ensure we don't go past the text length
      if (start >= text.length - overlap) {
        break;
      }
    }

    return chunks;
  }

  /**
   * Health check for Ollama service
   */
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      return response.ok;
    } catch (error: any) {
      console.error('Ollama health check failed:', error);
      return false;
    }
  }

  /**
   * Get available models from Ollama
   */
  async getAvailableModels(): Promise<string[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/tags`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch models');
      }

      const data = await response.json();
      return data.models?.map((model: any) => model.name) || [];
    } catch (error: any) {
      console.error('Error fetching available models:', error);
      return [];
    }
  }
}

// Create singleton instance
export const embeddingService = new EmbeddingService();
;
// Types are already exported as interfaces above