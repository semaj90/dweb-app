/**
 * Enhanced Vector Operations with pgvector + Ollama + NVIDIA go-llama
 * Integrates PostgreSQL vector search with AI services
 */

import { db } from './index.js';
import { cases, evidence, criminals } from './schema-postgres-enhanced.js';
import { and, desc, sql, gt, lt } from 'drizzle-orm';
import { productionServiceClient } from '../../services/productionServiceClient.js';

export interface VectorSearchResult {
  id: string;
  content: string;
  similarity: number;
  metadata: any;
  embedding?: number[];
}

export interface RAGContext {
  query: string;
  userId: string;
  caseId?: string;
  limit?: number;
  threshold?: number;
  includeMetadata?: boolean;
}

export class EnhancedVectorOperations {
  private readonly EMBEDDING_DIMENSIONS = 768; // nomic-embed-text dimensions
  private readonly DEFAULT_THRESHOLD = 0.7;
  private readonly DEFAULT_LIMIT = 10;

  /**
   * Generate embedding using Ollama/NVIDIA go-llama
   */
  async generateEmbedding(text: string, model: 'nomic-embed-text' | 'nvidia-llama' = 'nomic-embed-text'): Promise<number[]> {
    try {
      // Use production service client to get embedding
      const response = await productionServiceClient.execute('embedding.generate', {
        text,
        model,
        dimensions: this.EMBEDDING_DIMENSIONS
      });

      return response.embedding || response.data?.embedding || [];
    } catch (error) {
      console.error('Embedding generation failed:', error);
      // Fallback to mock embedding for development
      return new Array(this.EMBEDDING_DIMENSIONS).fill(0).map(() => Math.random() - 0.5);
    }
  }

  /**
   * Enhanced RAG search with multi-table vector similarity
   */
  async performRAGSearch(context: RAGContext): Promise<VectorSearchResult[]> {
    const { query, userId, caseId, limit = this.DEFAULT_LIMIT, threshold = this.DEFAULT_THRESHOLD } = context;

    // Generate query embedding
    const queryEmbedding = await this.generateEmbedding(query);
    const embeddingVector = `[${queryEmbedding.join(',')}]`;

    try {
      // Multi-table vector search with PostgreSQL pgvector
      const searchResults = await db.execute(sql`
        WITH vector_search AS (
          -- Search in cases
          SELECT 
            'case' as source_type,
            id::text,
            title as content,
            1 - (content_embedding <=> ${embeddingVector}::vector) as similarity,
            jsonb_build_object(
              'type', 'case',
              'case_number', case_number,
              'status', status,
              'category', category,
              'danger_score', danger_score,
              'created_at', created_at
            ) as metadata
          FROM cases 
          WHERE user_id = ${userId}
            ${caseId ? sql`AND id = ${caseId}` : sql``}
            AND content_embedding IS NOT NULL
            AND 1 - (content_embedding <=> ${embeddingVector}::vector) > ${threshold}

          UNION ALL

          -- Search in documents
          SELECT 
            'document' as source_type,
            id::text,
            COALESCE(title, filename) as content,
            1 - (content_embedding <=> ${embeddingVector}::vector) as similarity,
            jsonb_build_object(
              'type', 'document',
              'filename', filename,
              'file_type', file_type,
              'case_id', case_id,
              'upload_date', upload_date,
              'file_size', file_size
            ) as metadata
          FROM documents 
          WHERE user_id = ${userId}
            ${caseId ? sql`AND case_id = ${caseId}` : sql``}
            AND content_embedding IS NOT NULL
            AND 1 - (content_embedding <=> ${embeddingVector}::vector) > ${threshold}

          UNION ALL

          -- Search in evidence
          SELECT 
            'evidence' as source_type,
            id::text,
            description as content,
            1 - (description_embedding <=> ${embeddingVector}::vector) as similarity,
            jsonb_build_object(
              'type', 'evidence',
              'evidence_type', evidence_type,
              'case_id', case_id,
              'chain_of_custody', chain_of_custody,
              'collected_date', collected_date,
              'location', location
            ) as metadata
          FROM evidence 
          WHERE user_id = ${userId}
            ${caseId ? sql`AND case_id = ${caseId}` : sql``}
            AND description_embedding IS NOT NULL
            AND 1 - (description_embedding <=> ${embeddingVector}::vector) > ${threshold}
        )
        SELECT * FROM vector_search 
        ORDER BY similarity DESC 
        LIMIT ${limit}
      `);

      return searchResults.map((row: any) => ({
        id: String(row.id),
        content: String(row.content),
        similarity: parseFloat(String(row.similarity)),
        metadata: row.metadata,
        sourceType: row.source_type
      }));

    } catch (error) {
      console.error('RAG search failed:', error);
      return [];
    }
  }

  /**
   * Semantic case clustering using Neo4j-style graph operations
   */
  async findSimilarCases(caseId: string, userId: string, limit: number = 5): Promise<VectorSearchResult[]> {
    try {
      // Get the target case embedding
      const targetCase = await db.select({
        contentEmbedding: cases.contentEmbedding,
        title: cases.title
      }).from(cases).where(and(
        sql`id = ${caseId}`,
        sql`user_id = ${userId}`
      )).limit(1);

      if (!targetCase.length || !targetCase[0].contentEmbedding) {
        return [];
      }

      const embeddingVector = targetCase[0].contentEmbedding;

      // Find similar cases using cosine similarity
      const similarCases = await db.execute(sql`
        SELECT 
          id::text,
          title as content,
          1 - (content_embedding <=> ${embeddingVector}::vector) as similarity,
          jsonb_build_object(
            'case_number', case_number,
            'status', status,
            'category', category,
            'priority', priority,
            'danger_score', danger_score,
            'estimated_value', estimated_value,
            'jurisdiction', jurisdiction,
            'created_at', created_at
          ) as metadata
        FROM cases 
        WHERE user_id = ${userId}
          AND id != ${caseId}
          AND content_embedding IS NOT NULL
        ORDER BY content_embedding <=> ${embeddingVector}::vector
        LIMIT ${limit}
      `);

      return similarCases.map((row: any) => ({
        id: String(row.id),
        content: String(row.content),
        similarity: parseFloat(String(row.similarity)),
        metadata: row.metadata
      }));

    } catch (error) {
      console.error('Similar cases search failed:', error);
      return [];
    }
  }

  /**
   * Multi-core Ollama cluster query with load balancing
   */
  async enhancedRAGQuery(query: string, context: VectorSearchResult[], userId: string): Promise<{
    response: string;
    sources: VectorSearchResult[];
    model: string;
    processingTime: number;
  }> {
    const startTime = Date.now();

    try {
      // Prepare context for RAG
      const contextText = context.map(item => 
        `Source: ${item.metadata?.type || 'unknown'}\n` +
        `Content: ${item.content}\n` +
        `Relevance: ${(item.similarity * 100).toFixed(1)}%\n`
      ).join('\n---\n');

      // Enhanced RAG query using production service
      const response = await productionServiceClient.execute('rag.enhanced_query', {
        query,
        context: contextText,
        userId,
        options: {
          model: 'gemma3-legal',
          temperature: 0.7,
          max_tokens: 1000,
          use_multicore: true,
          include_sources: true
        }
      });

      return {
        response: response.answer || response.response || 'No response generated',
        sources: context,
        model: response.model || 'gemma3-legal',
        processingTime: Date.now() - startTime
      };

    } catch (error) {
      console.error('Enhanced RAG query failed:', error);
      return {
        response: 'Sorry, I encountered an error processing your query.',
        sources: context,
        model: 'error',
        processingTime: Date.now() - startTime
      };
    }
  }

  /**
   * Update embeddings for new/modified content
   */
  async updateContentEmbeddings(contentId: string, content: string, table: 'cases' | 'documents' | 'evidence'): Promise<void> {
    try {
      const embedding = await this.generateEmbedding(content);
      const embeddingVector = `[${embedding.join(',')}]`;

      switch (table) {
        case 'cases':
          await db.update(cases)
            .set({ 
              contentEmbedding: sql`${embeddingVector}::vector`,
              updatedAt: new Date()
            })
            .where(sql`id = ${contentId}`);
          break;

        // documents table not implemented yet

        case 'evidence':
          await db.update(evidence)
            .set({ 
              contentEmbedding: sql`${embeddingVector}::vector`,
              updatedAt: new Date()
            })
            .where(sql`id = ${contentId}`);
          break;
      }

    } catch (error) {
      console.error('Failed to update embeddings:', error);
      throw error;
    }
  }

  /**
   * Batch embedding generation for existing content
   */
  async batchGenerateEmbeddings(batchSize: number = 10): Promise<{ processed: number; errors: number }> {
    let processed = 0;
    let errors = 0;

    try {
      // Process cases without embeddings
      const casesWithoutEmbeddings = await db.select({
        id: cases.id,
        title: cases.title,
        description: cases.description
      }).from(cases).where(sql`content_embedding IS NULL`).limit(batchSize);

      for (const case_ of casesWithoutEmbeddings) {
        try {
          const content = `${case_.title}\n${case_.description}`;
          await this.updateContentEmbeddings(case_.id, content, 'cases');
          processed++;
        } catch (error) {
          console.error(`Failed to process case ${case_.id}:`, error);
          errors++;
        }
      }

      return { processed, errors };

    } catch (error) {
      console.error('Batch embedding generation failed:', error);
      return { processed, errors: errors + 1 };
    }
  }
}

// Singleton instance
export const vectorOps = new EnhancedVectorOperations();