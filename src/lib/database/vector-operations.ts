// PostgreSQL Vector Operations Service
// Optimized vector search and similarity operations using pgvector

import { db, queryClient } from './postgres-enhanced.js';
import { legalDocuments, legalEntities } from './schema/legal-documents.js';
import { sql, desc, asc, and, or, eq, ilike, inArray } from 'drizzle-orm';
import type { LegalDocument, LegalEntity } from './schema/legal-documents.js';

export interface VectorSearchOptions {
  limit?: number;
  threshold?: number;
  filter?: {
    documentType?: string[];
    jurisdiction?: string[];
    practiceArea?: string[];
    isConfidential?: boolean;
    dateRange?: {
      start: Date;
      end: Date;
    };
  };
}

export interface SearchResult<T> {
  item: T;
  similarity: number;
  rank: number;
}

/**
 * Vector Search Service for Legal Documents
 * Provides semantic search capabilities using pgvector
 */
export class VectorSearchService {
  /**
   * Search documents by content similarity
   */
  async searchDocuments(
    queryEmbedding: number[],
    options: VectorSearchOptions = {}
  ): Promise<SearchResult<LegalDocument>[]> {
    const {
      limit = 10,
      threshold = 0.7,
      filter = {}
    } = options;

    // Build filter conditions
    const filterConditions = [];
    
    if (filter.documentType && filter.documentType.length > 0) {
      filterConditions.push(inArray(legalDocuments.documentType, filter.documentType));
    }
    
    if (filter.jurisdiction && filter.jurisdiction.length > 0) {
      filterConditions.push(inArray(legalDocuments.jurisdiction, filter.jurisdiction));
    }
    
    if (filter.practiceArea && filter.practiceArea.length > 0) {
      filterConditions.push(inArray(legalDocuments.practiceArea, filter.practiceArea));
    }
    
    if (filter.isConfidential !== undefined) {
      filterConditions.push(eq(legalDocuments.isConfidential, filter.isConfidential));
    }
    
    if (filter.dateRange) {
      filterConditions.push(
        and(
          sql`${legalDocuments.createdAt} >= ${filter.dateRange.start}`,
          sql`${legalDocuments.createdAt} <= ${filter.dateRange.end}`
        )
      );
    }

    // Perform vector similarity search
    const results = await db
      .select({
        document: legalDocuments,
        similarity: sql<number>`1 - (${legalDocuments.contentEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`.as('similarity')
      })
      .from(legalDocuments)
      .where(
        and(
          sql`${legalDocuments.contentEmbedding} IS NOT NULL`,
          sql`1 - (${legalDocuments.contentEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector) >= ${threshold}`,
          ...(filterConditions.length > 0 ? [and(...filterConditions)] : [])
        )
      )
      .orderBy(desc(sql`1 - (${legalDocuments.contentEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`))
      .limit(limit);

    return results.map((result, index) => ({
      item: result.document,
      similarity: result.similarity,
      rank: index + 1
    }));
  }

  /**
   * Search documents by title similarity
   */
  async searchDocumentsByTitle(
    queryEmbedding: number[],
    options: VectorSearchOptions = {}
  ): Promise<SearchResult<LegalDocument>[]> {
    const {
      limit = 10,
      threshold = 0.7,
      filter = {}
    } = options;

    // Build filter conditions (same as above)
    const filterConditions = [];
    
    if (filter.documentType && filter.documentType.length > 0) {
      filterConditions.push(inArray(legalDocuments.documentType, filter.documentType));
    }
    
    if (filter.jurisdiction && filter.jurisdiction.length > 0) {
      filterConditions.push(inArray(legalDocuments.jurisdiction, filter.jurisdiction));
    }

    const results = await db
      .select({
        document: legalDocuments,
        similarity: sql<number>`1 - (${legalDocuments.titleEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`.as('similarity')
      })
      .from(legalDocuments)
      .where(
        and(
          sql`${legalDocuments.titleEmbedding} IS NOT NULL`,
          sql`1 - (${legalDocuments.titleEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector) >= ${threshold}`,
          ...(filterConditions.length > 0 ? [and(...filterConditions)] : [])
        )
      )
      .orderBy(desc(sql`1 - (${legalDocuments.titleEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`))
      .limit(limit);

    return results.map((result, index) => ({
      item: result.document,
      similarity: result.similarity,
      rank: index + 1
    }));
  }

  /**
   * Hybrid search combining content and title similarity
   */
  async hybridSearch(
    contentEmbedding: number[],
    titleEmbedding: number[],
    options: VectorSearchOptions & { 
      contentWeight?: number; 
      titleWeight?: number;
    } = {}
  ): Promise<SearchResult<LegalDocument>[]> {
    const {
      limit = 10,
      threshold = 0.7,
      contentWeight = 0.7,
      titleWeight = 0.3,
      filter = {}
    } = options;

    // Build filter conditions
    const filterConditions = [];
    
    if (filter.documentType && filter.documentType.length > 0) {
      filterConditions.push(inArray(legalDocuments.documentType, filter.documentType));
    }

    const results = await db
      .select({
        document: legalDocuments,
        contentSimilarity: sql<number>`1 - (${legalDocuments.contentEmbedding} <=> ${JSON.stringify(contentEmbedding)}::vector)`.as('content_similarity'),
        titleSimilarity: sql<number>`1 - (${legalDocuments.titleEmbedding} <=> ${JSON.stringify(titleEmbedding)}::vector)`.as('title_similarity'),
        hybridScore: sql<number>`(${contentWeight} * (1 - (${legalDocuments.contentEmbedding} <=> ${JSON.stringify(contentEmbedding)}::vector)) + ${titleWeight} * (1 - (${legalDocuments.titleEmbedding} <=> ${JSON.stringify(titleEmbedding)}::vector)))`.as('hybrid_score')
      })
      .from(legalDocuments)
      .where(
        and(
          sql`${legalDocuments.contentEmbedding} IS NOT NULL`,
          sql`${legalDocuments.titleEmbedding} IS NOT NULL`,
          sql`(${contentWeight} * (1 - (${legalDocuments.contentEmbedding} <=> ${JSON.stringify(contentEmbedding)}::vector)) + ${titleWeight} * (1 - (${legalDocuments.titleEmbedding} <=> ${JSON.stringify(titleEmbedding)}::vector))) >= ${threshold}`,
          ...(filterConditions.length > 0 ? [and(...filterConditions)] : [])
        )
      )
      .orderBy(desc(sql`(${contentWeight} * (1 - (${legalDocuments.contentEmbedding} <=> ${JSON.stringify(contentEmbedding)}::vector)) + ${titleWeight} * (1 - (${legalDocuments.titleEmbedding} <=> ${JSON.stringify(titleEmbedding)}::vector)))`))
      .limit(limit);

    return results.map((result, index) => ({
      item: result.document,
      similarity: result.hybridScore,
      rank: index + 1
    }));
  }

  /**
   * Search entities by name similarity
   */
  async searchEntities(
    queryEmbedding: number[],
    options: { limit?: number; threshold?: number; entityType?: string[] } = {}
  ): Promise<SearchResult<LegalEntity>[]> {
    const { limit = 10, threshold = 0.7, entityType = [] } = options;

    const filterConditions = [];
    if (entityType.length > 0) {
      filterConditions.push(inArray(legalEntities.entityType, entityType));
    }

    const results = await db
      .select({
        entity: legalEntities,
        similarity: sql<number>`1 - (${legalEntities.nameEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`.as('similarity')
      })
      .from(legalEntities)
      .where(
        and(
          sql`${legalEntities.nameEmbedding} IS NOT NULL`,
          sql`1 - (${legalEntities.nameEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector) >= ${threshold}`,
          eq(legalEntities.isActive, true),
          ...(filterConditions.length > 0 ? [and(...filterConditions)] : [])
        )
      )
      .orderBy(desc(sql`1 - (${legalEntities.nameEmbedding} <=> ${JSON.stringify(queryEmbedding)}::vector)`))
      .limit(limit);

    return results.map((result, index) => ({
      item: result.entity,
      similarity: result.similarity,
      rank: index + 1
    }));
  }

  /**
   * Find similar documents to a given document
   */
  async findSimilarDocuments(
    documentId: string,
    options: VectorSearchOptions = {}
  ): Promise<SearchResult<LegalDocument>[]> {
    const { limit = 5, threshold = 0.6 } = options;

    // Get the source document's embedding
    const sourceDoc = await db
      .select({
        contentEmbedding: legalDocuments.contentEmbedding
      })
      .from(legalDocuments)
      .where(eq(legalDocuments.id, documentId))
      .limit(1);

    if (!sourceDoc[0] || !sourceDoc[0].contentEmbedding) {
      return [];
    }

    // Find similar documents (excluding the source document)
    const results = await db
      .select({
        document: legalDocuments,
        similarity: sql<number>`1 - (${legalDocuments.contentEmbedding} <=> ${sourceDoc[0].contentEmbedding})`.as('similarity')
      })
      .from(legalDocuments)
      .where(
        and(
          sql`${legalDocuments.contentEmbedding} IS NOT NULL`,
          sql`${legalDocuments.id} != ${documentId}`,
          sql`1 - (${legalDocuments.contentEmbedding} <=> ${sourceDoc[0].contentEmbedding}) >= ${threshold}`
        )
      )
      .orderBy(desc(sql`1 - (${legalDocuments.contentEmbedding} <=> ${sourceDoc[0].contentEmbedding})`))
      .limit(limit);

    return results.map((result, index) => ({
      item: result.document,
      similarity: result.similarity,
      rank: index + 1
    }));
  }

  /**
   * Batch update embeddings for documents
   */
  async updateDocumentEmbeddings(
    updates: Array<{
      id: string;
      contentEmbedding?: number[];
      titleEmbedding?: number[];
    }>
  ): Promise<void> {
    // Use transaction for batch updates
    await queryClient.begin(async (sql) => {
      for (const update of updates) {
        const setClause: unknown = {};
        
        if (update.contentEmbedding) {
          setClause.contentEmbedding = JSON.stringify(update.contentEmbedding);
        }
        
        if (update.titleEmbedding) {
          setClause.titleEmbedding = JSON.stringify(update.titleEmbedding);
        }

        if (Object.keys(setClause).length > 0) {
          setClause.updatedAt = new Date();
          
          await sql`
            UPDATE legal_documents 
            SET ${sql(setClause)}
            WHERE id = ${update.id}
          `;
        }
      }
    });
  }

  /**
   * Get vector statistics for monitoring
   */
  async getVectorStats(): Promise<{
    totalDocuments: number;
    documentsWithContentEmbeddings: number;
    documentsWithTitleEmbeddings: number;
    averageContentEmbeddingNorm: number;
    averageTitleEmbeddingNorm: number;
  }> {
    const stats = await queryClient`
      SELECT 
        COUNT(*) as total_documents,
        COUNT(content_embedding) as documents_with_content_embeddings,
        COUNT(title_embedding) as documents_with_title_embeddings,
        AVG(array_length(content_embedding::float[], 1)) as avg_content_dim,
        AVG(array_length(title_embedding::float[], 1)) as avg_title_dim
      FROM legal_documents
    `;

    const norms = await queryClient`
      SELECT 
        AVG(sqrt(array_dot(content_embedding::float[], content_embedding::float[]))) as avg_content_norm,
        AVG(sqrt(array_dot(title_embedding::float[], title_embedding::float[]))) as avg_title_norm
      FROM legal_documents 
      WHERE content_embedding IS NOT NULL AND title_embedding IS NOT NULL
    `;

    return {
      totalDocuments: parseInt(stats[0].total_documents),
      documentsWithContentEmbeddings: parseInt(stats[0].documents_with_content_embeddings),
      documentsWithTitleEmbeddings: parseInt(stats[0].documents_with_title_embeddings),
      averageContentEmbeddingNorm: parseFloat(norms[0]?.avg_content_norm || '0'),
      averageTitleEmbeddingNorm: parseFloat(norms[0]?.avg_title_norm || '0')
    };
  }

  /**
   * Create vector index if not exists
   */
  async ensureVectorIndexes(): Promise<void> {
    try {
      // Create HNSW indexes for better performance
      await queryClient`
        CREATE INDEX CONCURRENTLY IF NOT EXISTS legal_documents_content_embedding_hnsw_idx 
        ON legal_documents USING hnsw (content_embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64)
      `;

      await queryClient`
        CREATE INDEX CONCURRENTLY IF NOT EXISTS legal_documents_title_embedding_hnsw_idx 
        ON legal_documents USING hnsw (title_embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64)
      `;

      await queryClient`
        CREATE INDEX CONCURRENTLY IF NOT EXISTS legal_entities_name_embedding_hnsw_idx 
        ON legal_entities USING hnsw (name_embedding vector_cosine_ops) 
        WITH (m = 16, ef_construction = 64)
      `;

      console.log('Vector indexes ensured successfully');
    } catch (error) {
      console.error('Error ensuring vector indexes:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const vectorSearchService = new VectorSearchService();

// Utility functions for embedding operations
export const embeddingUtils = {
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

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  },

  /**
   * Normalize vector to unit length
   */
  normalizeVector(vector: number[]): number[] {
    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return norm === 0 ? vector : vector.map(val => val / norm);
  },

  /**
   * Validate embedding dimensions
   */
  validateEmbedding(embedding: number[], expectedDim: number = 384): boolean {
    return Array.isArray(embedding) && 
           embedding.length === expectedDim && 
           embedding.every(val => typeof val === 'number' && !isNaN(val));
  }
};