// @ts-nocheck
import { db } from "$lib/server/database";
// Orphaned content: import {
legalDocuments as documents,
  embeddingCache,
import { eq, sql, desc, asc } from "drizzle-orm";
import crypto from "crypto";

export interface VectorSearchOptions {
  threshold?: number;
  limit?: number;
  caseId?: string;
  documentType?: string;
}

export interface VectorSearchResult {
  id: string;
  content: string;
  filename?: string;
  caseId?: string;
  distance: number;
  relevanceScore: number;
  summary?: string;
  keywords?: string[];
  createdAt: Date;
}

export interface EmbeddingCacheEntry {
  textHash: string;
  embedding: number[];
  model: string;
  dimensions: number;
}

export class VectorSearchService {
  private static instance: VectorSearchService;

  public static getInstance(): VectorSearchService {
    if (!VectorSearchService.instance) {
      VectorSearchService.instance = new VectorSearchService();
    }
    return VectorSearchService.instance;
  }

  /**
   * Generate text hash for embedding cache
   */
  private generateTextHash(text: string): string {
    return crypto
      .createHash("sha256")
      .update(text.trim().toLowerCase())
      .digest("hex");
  }

  /**
   * Get or create embedding with caching
   */
  private async getOrCreateEmbedding(
    text: string,
    model: string = "ollama-nomic-embed-text"
  ): Promise<number[]> {
    const textHash = this.generateTextHash(text);

    // Check cache first
    const cached = await db
      .select()
      .from(embeddingCache)
      .where(eq(embeddingCache.textHash, textHash))
      .limit(1);

    if (cached.length > 0) {
      return cached[0].embedding as number[];
    }

    // Generate new embedding
    let embedding: number[];

    try {
      if (model.startsWith("ollama-")) {
        embedding = await this.generateOllamaEmbedding(
          text,
          model.replace("ollama-", "")
        );
      } else if (model.startsWith("claude-")) {
        // Claude doesn't have embeddings API, fallback to Ollama
        embedding = await this.generateOllamaEmbedding(
          text,
          "nomic-embed-text"
        );
      } else if (model.startsWith("gemini-")) {
        embedding = await this.generateGeminiEmbedding(text);
      } else {
        // Default to Ollama
        embedding = await this.generateOllamaEmbedding(
          text,
          "nomic-embed-text"
        );
      }

      // Cache the embedding
      await db
        .insert(embeddingCache)
        .values({
          textHash,
          embedding,
          model,
          createdAt: new Date(),
        })
        .onConflictDoNothing();

      return embedding;
    } catch (error) {
      console.error("Failed to generate embedding:", error);
      throw error;
    }
  }

  /**
   * Generate embedding using Ollama
   */
  private async generateOllamaEmbedding(
    text: string,
    model: string = "nomic-embed-text"
  ): Promise<number[]> {
    const response = await fetch("http://localhost:11434/api/embeddings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        prompt: text,
      }),
    });

    if (!response.ok) {
      throw new Error(`Ollama embedding failed: ${response.statusText}`);
    }

    const data = await response.json();
    return data.embedding;
  }

  /**
   * Generate embedding using Gemini
   */
  private async generateGeminiEmbedding(text: string): Promise<number[]> {
    // Implement Gemini embedding API call
    // This would require the Gemini API key and proper setup
    throw new Error("Gemini embeddings not yet implemented");
  }

  /**
   * Perform vector similarity search
   */
  async search(
    query: string,
    options: VectorSearchOptions = {}
  ): Promise<VectorSearchResult[]> {
    const { threshold = 0.7, limit = 10, caseId, documentType } = options;

    try {
      // Generate query embedding
      const queryEmbedding = await this.getOrCreateEmbedding(query);
      const embeddingVector = `[${queryEmbedding.join(",")}]`;

      // Build SQL query
      let sqlQuery = sql`
        SELECT
          d.id,
          d.extracted_text as content,
          d.filename,
          d.case_id,
          d.summary,
          d.keywords,
          d.created_at,
          (d.embedding <=> ${embeddingVector}::vector) as distance,
          (1 - (d.embedding <=> ${embeddingVector}::vector)) as relevance_score
        FROM ${documents} d
        WHERE d.embedding IS NOT NULL
      `;

      // Add filters
      const conditions = [];

      if (caseId) {
        conditions.push(sql`d.case_id = ${caseId}`);
      }

      if (documentType) {
        conditions.push(sql`d.document_type = ${documentType}`);
      }

      // Add similarity threshold
      conditions.push(
        sql`(d.embedding <=> ${embeddingVector}::vector) < ${1 - threshold}`
      );

      if (conditions.length > 0) {
        sqlQuery = sql`${sqlQuery} AND ${sql.join(conditions, sql` AND `)}`;
      }

      // Order by similarity and limit
      sqlQuery = sql`
        ${sqlQuery}
        ORDER BY distance ASC
        LIMIT ${limit}
      `;

      const results = await db.execute(sqlQuery);

      return results.rows.map((row: any) => ({
        id: row.id,
        content: row.content,
        filename: row.filename,
        caseId: row.case_id,
        distance: parseFloat(row.distance),
        relevanceScore: parseFloat(row.relevance_score),
        summary: row.summary,
        keywords: row.keywords,
        createdAt: new Date(row.created_at),
      }));
    } catch (error) {
      console.error("Vector search failed:", error);
      throw error;
    }
  }

  /**
   * Index a document for vector search
   */
  async indexDocument(
    documentId: string,
    content: string,
    metadata: {
      filename?: string;
      caseId?: string;
      documentType?: string;
      summary?: string;
      keywords?: string[];
    }
  ): Promise<void> {
    try {
      // Generate embedding for document content
      const embedding = await this.getOrCreateEmbedding(content);

      // Update document with embedding
      await db
        .update(documents)
        .set({
          embedding: embedding as any, // Drizzle will handle vector type conversion
          summary: metadata.summary,
          keywords: metadata.keywords,
          updatedAt: new Date(),
        })
        .where(eq(documents.id, documentId));
    } catch (error) {
      console.error("Document indexing failed:", error);
      throw error;
    }
  }

  /**
   * Get embedding cache statistics
   */
  async getCacheStats() {
    const stats = await db.execute(sql`
      SELECT
        COUNT(*) as total_embeddings,
        COUNT(DISTINCT model) as unique_models,
        AVG(dimensions) as avg_dimensions,
        MIN(created_at) as oldest_embedding,
        MAX(created_at) as newest_embedding
      FROM ${embeddingCache}
    `);

    return stats.rows[0];
  }

  /**
   * Create PostgreSQL index for vector similarity search
   */
  async createVectorIndex(): Promise<void> {
    try {
      // Create IVFFlat index for better performance
      await db.execute(sql`
        CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_documents_embedding_ivfflat
        ON documents USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100)
      `);

      // Analyze table for better query planning
      await db.execute(sql`ANALYZE documents`);

      console.log("Vector index created successfully");
    } catch (error) {
      console.error("Failed to create vector index:", error);
      throw error;
    }
  }

  /**
   * Build context for Claude/Gemini with vector search results
   */
  async buildLegalContext(
    query: string,
    caseId?: string,
    maxContext: number = 5000
  ): Promise<{
    context: string;
    sources: VectorSearchResult[];
    relevanceScores: number[];
  }> {
    const searchResults = await this.search(query, {
      caseId,
      threshold: 0.6,
      limit: 10,
    });

    // Sort by relevance and build context
    const topResults = searchResults
      .sort((a, b) => b.relevanceScore - a.relevanceScore)
      .slice(0, 5);

    let context = "";
    let currentLength = 0;
    const usedSources: VectorSearchResult[] = [];
    const relevanceScores: number[] = [];

    for (const result of topResults) {
      const resultText = `Document: ${result.filename || "Unknown"}\nContent: ${result.content}\nSummary: ${result.summary || "No summary"}\n\n`;

      if (currentLength + resultText.length <= maxContext) {
        context += resultText;
        currentLength += resultText.length;
        usedSources.push(result);
        relevanceScores.push(result.relevanceScore);
      } else {
        break;
      }
    }

    return {
      context,
      sources: usedSources,
      relevanceScores,
    };
  }
}

// Export singleton instance
export const vectorSearchService = VectorSearchService.getInstance();
